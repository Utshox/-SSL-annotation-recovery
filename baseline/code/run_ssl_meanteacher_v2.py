
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import argparse
import json
from pathlib import Path

# --- Configuration ---
IMG_SIZE = 256
BATCH_SIZE = 16 # (4 labeled + 4 unlabeled)
EPOCHS = 100
LR = 1e-4
ALPHA = 0.999 # EMA decay for teacher

# --- U-Net ---
def get_unet(img_size=256, num_classes=1, dropout=False):
    inputs = layers.Input(shape=(img_size, img_size, 1))

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        if dropout:
            x = layers.Dropout(0.1)(x, training=True) # MC Dropout
        return x

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = conv_block(p4, 512)

    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 256)
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 128)
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 64)
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 32)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    return models.Model(inputs=[inputs], outputs=[outputs])

# --- Rampup Function ---
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

# --- Mean Teacher Trainer ---
class MeanTeacherTrainer(models.Model):
    def __init__(self, student, teacher, alpha=0.999, max_lambda=10.0, rampup_epochs=40):
        super(MeanTeacherTrainer, self).__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.max_lambda = max_lambda
        self.rampup_epochs = rampup_epochs
        self.epoch_tracker = tf.Variable(0, trainable=False, dtype=tf.int32)
        
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()

    def update_teacher(self):
        for s_var, t_var in zip(self.student.variables, self.teacher.variables):
            t_var.assign(self.alpha * t_var + (1.0 - self.alpha) * s_var)

    def train_step(self, data):
        (x_l, y_l), x_u = data
        
        # Current consistency weight
        current_epoch = tf.cast(self.epoch_tracker, tf.float32)
        # We can't use numpy inside train_step easily without tf.py_function
        # Use a simple TF-based rampup or just pass it in.
        # For simplicity, let's use a linear rampup in TF
        rampup_val = tf.minimum(1.0, current_epoch / float(self.rampup_epochs))
        consistency_weight = self.max_lambda * tf.exp(-5.0 * tf.pow(1.0 - rampup_val, 2))
        
        # Add noise to student inputs
        x_l_noise = x_l + tf.random.normal(tf.shape(x_l), 0, 0.05)
        x_u_noise = x_u + tf.random.normal(tf.shape(x_u), 0, 0.05)
        
        with tf.GradientTape() as tape:
            # 1. Supervised Loss
            y_l_pred = self.student(x_l_noise, training=True)
            loss_sup = self.bce(y_l, y_l_pred)
            
            # 2. Consistency Loss
            y_u_student = self.student(x_u_noise, training=True)
            y_u_teacher = self.teacher(x_u, training=False)
            loss_cons = self.mse(y_u_teacher, y_u_student)
            
            total_loss = loss_sup + consistency_weight * loss_cons

        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        
        self.update_teacher()
        
        self.compiled_metrics.update_state(y_l, y_l_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss_sup, "loss_cons": loss_cons, "lambda": consistency_weight})
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.teacher(x, training=False)
        loss = self.bce(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# --- Data Generator ---

def load_and_concat(file_list):
    arrs = [np.load(f) for f in file_list if os.path.exists(f)]
    if not arrs: return np.array([])
    return np.concatenate(arrs, axis=0)

class DualGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_l, y_l, x_u, batch_size=16):
        self.x_l = x_l
        self.y_l = y_l
        self.x_u = x_u
        self.batch_size = batch_size
        self.half_bs = batch_size // 2

    def __len__(self):
        return len(self.x_l) // self.half_bs

    def __getitem__(self, idx):
        inds_l = np.random.choice(len(self.x_l), self.half_bs)
        batch_xl = self.x_l[inds_l]
        batch_yl = self.y_l[inds_l]
        
        inds_u = np.random.choice(len(self.x_u), self.half_bs)
        batch_xu = self.x_u[inds_u]
        
        batch_xl = np.expand_dims(batch_xl, -1).astype(np.float32)
        batch_yl = np.expand_dims(batch_yl, -1).astype(np.float32)
        batch_yl = np.clip(batch_yl, 0, 1)
        batch_xu = np.expand_dims(batch_xu, -1).astype(np.float32)
        
        return (batch_xl, batch_yl), batch_xu
class EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.epoch_tracker.assign(epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split_json', type=str, required=True)
    parser.add_argument('--ratio', type=int, default=10) # 10, 25, or 50
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.split_json, 'r') as f:
        splits = json.load(f)
    
    labeled_ids = splits[f"labeled_{args.ratio}"]
    unlabeled_ids = splits[f"unlabeled_{args.ratio}"]
    val_ids = splits["validation"]
    
    xl_files = [os.path.join(args.data_dir, f"{pid}_x.npy") for pid in labeled_ids]
    yl_files = [os.path.join(args.data_dir, f"{pid}_y.npy") for pid in labeled_ids]
    xu_files = [os.path.join(args.data_dir, f"{pid}_x.npy") for pid in unlabeled_ids]
    
    xv_files = [os.path.join(args.data_dir, f"{pid}_x.npy") for pid in val_ids]
    yv_files = [os.path.join(args.data_dir, f"{pid}_y.npy") for pid in val_ids]
    
    # Load val into RAM for faster evaluation
    X_val = np.concatenate([np.load(f) for f in xv_files], axis=0)
    Y_val = np.concatenate([np.load(f) for f in yv_files], axis=0)
    X_val = np.expand_dims(X_val, -1).astype(np.float32)
    Y_val = np.expand_dims(Y_val, -1).astype(np.float32)
    Y_val = np.clip(Y_val, 0, 1)
    
    X_l = load_and_concat(xl_files)
    Y_l = load_and_concat(yl_files)
    X_u = load_and_concat(xu_files)
    train_gen = DualGenerator(X_l, Y_l, X_u, BATCH_SIZE)
    
    student = get_unet()
    teacher = get_unet()
    teacher.set_weights(student.get_weights())
    
    model = MeanTeacherTrainer(student, teacher)
    model.compile(
        optimizer=optimizers.Adam(LR),
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')]
    )
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Saving weights only for the wrapper model
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'model_best.h5'), 
                                 save_best_only=True, monitor='val_io_u', mode='max', save_weights_only=True)
    
    model.fit(
        train_gen,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        callbacks=[checkpoint, CSVLogger(os.path.join(args.output_dir, 'log.csv')), EpochCallback()]
    )

if __name__ == "__main__":
    main()
