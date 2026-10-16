
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
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

# --- U-Net ---
def get_unet(img_size=256, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, 1))

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
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

# --- CPS Trainer ---
class CPSTrainer(models.Model):
    def __init__(self, model_a, model_b, max_lambda=1.0, rampup_epochs=40):
        super(CPSTrainer, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.max_lambda = max_lambda
        self.rampup_epochs = rampup_epochs
        self.epoch_tracker = tf.Variable(0, trainable=False, dtype=tf.int32)
        
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()

    def train_step(self, data):
        (x_l, y_l), x_u = data
        
        current_epoch = tf.cast(self.epoch_tracker, tf.float32)
        rampup_val = tf.minimum(1.0, current_epoch / float(self.rampup_epochs))
        consistency_weight = self.max_lambda * tf.exp(-5.0 * tf.pow(1.0 - rampup_val, 2))
        
        with tf.GradientTape() as tape:
            # 1. Supervised Loss (Both models)
            y_l_a = self.model_a(x_l, training=True)
            y_l_b = self.model_b(x_l, training=True)
            loss_sup = self.bce(y_l, y_l_a) + self.bce(y_l, y_l_b)
            
            # 2. Cross-Pseudo Supervision Loss (Unlabeled)
            # Soft version: MSE between predictions
            y_u_a = self.model_a(x_u, training=True)
            y_u_b = self.model_b(x_u, training=True)
            
            # Use pseudo-labels (hard version usually preferred in CPS)
            # Let's use hard pseudo-labels
            pseudo_a = tf.cast(y_u_a > 0.5, tf.float32)
            pseudo_b = tf.cast(y_u_b > 0.5, tf.float32)
            
            loss_cps = self.bce(pseudo_b, y_u_a) + self.bce(pseudo_a, y_u_b)
            
            total_loss = loss_sup + consistency_weight * loss_cps

        trainable_vars = self.model_a.trainable_variables + self.model_b.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        self.compiled_metrics.update_state(y_l, y_l_a) # Track Model A performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss_sup, "loss_cps": loss_cps})
        return results

    def test_step(self, data):
        x, y = data
        # Average both models for inference
        y_a = self.model_a(x, training=False)
        y_b = self.model_b(x, training=False)
        y_pred = (y_a + y_b) / 2.0
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
    parser.add_argument('--ratio', type=int, default=10)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.split_json, 'r') as f:
        splits = json.load(f)
    
    xl_files = [os.path.join(args.data_dir, f"{pid}_x.npy") for pid in splits[f"labeled_{args.ratio}"]]
    yl_files = [os.path.join(args.data_dir, f"{pid}_y.npy") for pid in splits[f"labeled_{args.ratio}"]]
    xu_files = [os.path.join(args.data_dir, f"{pid}_x.npy") for pid in splits[f"unlabeled_{args.ratio}"]]
    xv_files = [os.path.join(args.data_dir, f"{pid}_x.npy") for pid in splits["validation"]]
    yv_files = [os.path.join(args.data_dir, f"{pid}_y.npy") for pid in splits["validation"]]
    
    X_val = np.concatenate([np.load(f) for f in xv_files], axis=0)
    Y_val = np.concatenate([np.load(f) for f in yv_files], axis=0)
    X_val = np.expand_dims(X_val, -1).astype(np.float32)
    Y_val = np.expand_dims(Y_val, -1).astype(np.float32)
    Y_val = np.clip(Y_val, 0, 1)
    
    X_l = load_and_concat(xl_files)
    Y_l = load_and_concat(yl_files)
    X_u = load_and_concat(xu_files)
    train_gen = DualGenerator(X_l, Y_l, X_u, BATCH_SIZE)
    
    model_a = get_unet()
    model_b = get_unet() # Different initialization
    
    model = CPSTrainer(model_a, model_b)
    model.compile(
        optimizer=optimizers.Adam(LR),
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')]
    )
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
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
