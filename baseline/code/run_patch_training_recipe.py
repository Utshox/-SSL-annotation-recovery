"""
Patch U-Net with an nnU-Net-style training recipe, 5-fold cross-validation
--------------------------------------------------------------------------
Third arm of the CBM gap decomposition. The 2.5D ablation showed inter-slice
context closes ~21% of the 0.623 -> 0.851 gap to nnU-Net 3d_fullres. This arm
asks how much of the remainder is not architecture at all but training recipe:

    plain arm   : binary cross-entropy, fixed LR 1e-4, no augmentation
    recipe arm  : Dice+BCE hybrid loss, cosine LR decay, spatial and intensity
                  augmentation

Everything else is imported from `run_patch_training_25d` (same U-Net, same
patch set, same KFold(n_splits=5, shuffle=True, random_state=42) patient split,
same 3D sliding-window inference, same volumetric Dice), so a difference
between arms is attributable to the recipe alone.

Run at --n_slices 1 and --n_slices 5 to complete the 2x2:

    n_slices=1, plain   0.623  (already measured, baseline/models/25d_1slice)
    n_slices=5, plain   0.670  (already measured, baseline/models/25d_5slice)
    n_slices=1, recipe  this script
    n_slices=5, recipe  this script

which separates the context effect, the recipe effect, and their interaction.

Usage:
    python run_patch_training_recipe.py --data_dir <v6 dir> --raw_dir <Task07> \
        --fold 0 --n_slices 5 --output_dir baseline/models/recipe_5slice
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

# Shared with the plain arms so the comparison stays controlled.
from run_patch_training_25d import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    LR,
    PatchSequence,
    get_patient_ids,
    get_unet,
    load_patient_patches,
    run_inference_on_volumes,
    set_seed,
)

# Patches are HU-clipped to [-125, 275] then min-max scaled, so intensity
# augmentation operates in [0, 1] and re-clips to that range.
VAL_MIN, VAL_MAX = 0.0, 1.0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_bce_loss(y_true, y_pred, smooth=1.0, w_bce=0.5, w_dice=0.5):
    """Equal-weight hybrid of BCE and soft Dice, Dice computed per sample.

    Plain BCE is dominated by background: the pancreas occupies well under 5% of
    a 256x256 patch even after balanced sampling, so the gradient is mostly
    telling the network to predict zero. The Dice term is scale-invariant with
    respect to foreground size and supplies the signal BCE cannot.
    """
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

    batch = tf.shape(y_true)[0]
    y_true_f = tf.reshape(y_true, (batch, -1))
    y_pred_f = tf.reshape(y_pred, (batch, -1))
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return w_bce * bce + w_dice * (1.0 - tf.reduce_mean(dice))


def _flat(y_true, y_pred, eps=1e-6):
    """Flatten to (batch, pixels) so overlap terms are computed per sample.

    y_pred is clipped away from 0 and 1. Without this the region losses drive the
    sigmoid into saturation within one epoch at this foreground ratio: once every
    prediction is exactly 0 the local gradient is 0, the smoothing term pins the
    loss at a constant, and training is dead for the rest of the run. That is
    what happened to the dice / tversky / focal_tversky arms on 2026-07-31
    (loss frozen at 5 decimal places from epoch 1 to epoch 99).
    """
    batch = tf.shape(y_true)[0]
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    return (tf.reshape(y_true, (batch, -1)),
            tf.reshape(y_pred, (batch, -1)))


def dice_loss(y_true, y_pred, smooth=1.0):
    """Soft Dice alone: fully scale-invariant, no per-pixel term at all.

    Isolates what the Dice half of the hybrid contributes. Pure Dice is known to
    give noisy gradients on samples whose foreground is empty or near-empty,
    which the BCE half of the hybrid otherwise damps.
    """
    y_true_f, y_pred_f = _flat(y_true, y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice)


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1.0):
    """Tversky index with false negatives weighted above false positives.

    TI = TP / (TP + alpha*FN + beta*FP). Dice is the special case alpha=beta=0.5.
    The plain BCE arm under-segments (it is biased toward predicting background
    at 2.1% foreground), so alpha=0.7 penalises missed pancreas more than spurious
    pancreas. Weights follow Salehi et al. (2017).
    """
    y_true_f, y_pred_f = _flat(y_true, y_pred)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f), axis=1)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f, axis=1)
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - tf.reduce_mean(ti)


def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
    """(1 - TI)^gamma, gamma < 1, so gradients grow on hard (low-TI) samples.

    Abraham & Khan (2019), proposed for exactly this regime: small lesions where
    easy background-dominated samples otherwise swamp the gradient.
    """
    y_true_f, y_pred_f = _flat(y_true, y_pred)
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f), axis=1)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f, axis=1)
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    # Guard the fractional power: float32 can leave ti a hair above 1.0, and
    # (negative) ** 0.75 is NaN. This produced loss=nan from epoch 0.
    return tf.reduce_mean(tf.pow(tf.maximum(1.0 - ti, 0.0) + 1e-8, gamma))


def focal_bce_loss(y_true, y_pred, gamma=2.0, alpha=0.75, eps=1e-7):
    """Focal loss (Lin et al., 2017): down-weights already-confident pixels.

    Keeps BCE's per-pixel character but removes the easy-background dominance
    that motivated the Dice term, so it separates "BCE is per-pixel" from
    "BCE is swamped by background". alpha=0.75 puts the larger weight on
    foreground, the segmentation convention, rather than RetinaNet's 0.25.
    """
    p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt = y_true * p + (1.0 - y_true) * (1.0 - p)
    a = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    return tf.reduce_mean(-a * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))


def weighted_bce_loss(y_true, y_pred, pos_weight=7.0, eps=1e-7):
    """BCE with the foreground term scaled up.

    Foreground occupies 2.07% of a v6 patch, a 47:1 imbalance. Full
    inverse-frequency weighting (47) is known to over-segment, so pos_weight is
    set to sqrt(47) ~= 7, the usual compromise. This is the minimal edit to BCE:
    same per-pixel form, only the class balance changes.
    """
    p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    return tf.reduce_mean(-(pos_weight * y_true * tf.math.log(p)
                            + (1.0 - y_true) * tf.math.log(1.0 - p)))


# Blend weight for the cross-entropy warm-up: 1.0 = pure BCE, 0.0 = pure target
# loss. A callback flips it at --warmup_epochs. Region losses need a sane
# starting point; started cold they collapse to all-background and never recover.
WARMUP_W = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="warmup_w")


def make_warmup_loss(target_fn):
    """Wrap a loss so it is BCE while WARMUP_W is 1 and the target once it is 0."""
    bce = tf.keras.losses.BinaryCrossentropy()

    def _loss(y_true, y_pred):
        return WARMUP_W * bce(y_true, y_pred) + (1.0 - WARMUP_W) * target_fn(y_true, y_pred)
    _loss.__name__ = f"warmup_{getattr(target_fn, '__name__', 'loss')}"
    return _loss


class WarmupSwitch(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        w = 1.0 if epoch < self.warmup_epochs else 0.0
        if float(WARMUP_W.numpy()) != w:
            print(f"\n[warmup] epoch {epoch}: switching loss weight to "
                  f"{'BCE' if w else 'target loss'}")
        WARMUP_W.assign(w)


# Registry: name -> loss passed to model.compile().
LOSSES = {
    "bce": "binary_crossentropy",
    "dice_bce": dice_bce_loss,
    "dice": dice_loss,
    "tversky": tversky_loss,
    "focal_tversky": focal_tversky_loss,
    "focal_bce": focal_bce_loss,
    "weighted_bce": weighted_bce_loss,
}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class AugmentedPatchSequence(PatchSequence):
    """PatchSequence plus spatial and intensity augmentation.

    Spatial transforms are drawn per sample and applied identically to the image
    stack and its mask. Intensity transforms are drawn once per batch, which
    costs far less than per-sample draws on a (32, 256, 256, C) array and still
    varies every step. Channels hold neighbouring slices, so in-plane transforms
    apply across all of them and slice order is left untouched.
    """

    def __init__(self, *args, rng_seed=1234, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_rng = np.random.default_rng(rng_seed)

    def __getitem__(self, i):
        # Fancy indexing in the parent already returns copies, safe to mutate.
        x, y = super().__getitem__(i)
        x = np.array(x, dtype=np.float32, copy=True)
        y = np.array(y, dtype=np.float32, copy=True)
        rng = self.aug_rng

        for n in range(x.shape[0]):
            if rng.random() < 0.5:                      # flip rows
                x[n] = x[n][::-1, :, :]
                y[n] = y[n][::-1, :, :]
            if rng.random() < 0.5:                      # flip columns
                x[n] = x[n][:, ::-1, :]
                y[n] = y[n][:, ::-1, :]
            if rng.random() < 0.5:                      # 90/180/270 rotation
                k = int(rng.integers(1, 4))
                x[n] = np.rot90(x[n], k=k, axes=(0, 1))
                y[n] = np.rot90(y[n], k=k, axes=(0, 1))

        if rng.random() < 0.20:                         # brightness
            x *= rng.uniform(0.75, 1.25)
        if rng.random() < 0.20:                         # gamma
            x = np.clip(x, VAL_MIN, VAL_MAX) ** rng.uniform(0.7, 1.5)
        if rng.random() < 0.20:                         # contrast about the mean
            mean = x.mean()
            x = (x - mean) * rng.uniform(0.75, 1.25) + mean
        if rng.random() < 0.15:                         # additive Gaussian noise
            x += rng.normal(0.0, rng.uniform(0.01, 0.05), size=x.shape).astype(np.float32)

        np.clip(x, VAL_MIN, VAL_MAX, out=x)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Patch U-Net with an nnU-Net-style training recipe, 5-fold CV.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="v6 2.5D patch directory")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--n_slices", type=int, default=5, choices=[1, 3, 5])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--no_augment", action="store_true",
                        help="Ablate the augmentation half of the recipe")
    parser.add_argument("--no_dice_loss", action="store_true",
                        help="Ablate the loss half of the recipe (keep plain BCE)")
    parser.add_argument("--no_cosine", action="store_true",
                        help="Ablate the LR schedule (keep fixed LR)")
    parser.add_argument("--loss", type=str, default=None, choices=sorted(LOSSES),
                        help="Loss function. Overrides --no_dice_loss. Default "
                             "keeps the original behaviour (dice_bce, or bce "
                             "when --no_dice_loss is set).")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Train with plain BCE for this many epochs before "
                             "switching to --loss. Region losses (dice, tversky, "
                             "focal_tversky) collapse to all-background within one "
                             "epoch if started cold at this foreground ratio; a "
                             "short BCE warm-up gives them a non-degenerate start. "
                             "0 reproduces the original (broken) behaviour.")
    args = parser.parse_args()

    # --loss wins; otherwise fall back to the original two-way flag so the
    # already-completed 2x2 runs stay reproducible from this script.
    loss_name = args.loss or ("bce" if args.no_dice_loss else "dice_bce")

    set_seed(args.seed)
    fold_dir = os.path.join(args.output_dir, f"fold_{args.fold}")
    Path(fold_dir).mkdir(parents=True, exist_ok=True)

    patient_ids = get_patient_ids(args.data_dir)
    print(f"Found {len(patient_ids)} patients in {args.data_dir}")

    # Same split as the plain arms -> folds are directly comparable.
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = list(kf.split(patient_ids))[args.fold]
    train_pids = [patient_ids[i] for i in train_idx]
    val_pids = [patient_ids[i] for i in val_idx]
    print(f"\nFold {args.fold}: {len(train_pids)} train, {len(val_pids)} val patients")

    recipe = {
        "loss": loss_name,
        "dice_bce_loss": loss_name == "dice_bce",
        "augmentation": not args.no_augment,
        "cosine_lr": not args.no_cosine,
    }
    print(f"Recipe: {recipe}")

    with open(os.path.join(fold_dir, "split_info.json"), "w") as f:
        json.dump({"fold": args.fold, "seed": args.seed, "n_slices": args.n_slices,
                   "recipe": recipe,
                   "n_train": len(train_pids), "n_val": len(val_pids),
                   "train_patients": train_pids, "val_patients": val_pids}, f, indent=2)

    print(f"\nLoading patches ({args.n_slices}-slice input)...")
    X_train, Y_train = load_patient_patches(args.data_dir, train_pids, args.n_slices)
    X_val, Y_val = load_patient_patches(args.data_dir, val_pids, args.n_slices)
    print(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")

    seq_cls = PatchSequence if args.no_augment else AugmentedPatchSequence
    train_seq = seq_cls(X_train, Y_train, args.batch_size, shuffle=True, seed=args.seed)
    val_seq = PatchSequence(X_val, Y_val, args.batch_size, shuffle=False)

    steps_per_epoch = len(train_seq)
    if args.no_cosine:
        lr_schedule = args.lr
    else:
        # Decay to 1% of the initial rate over the full run.
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.epochs * steps_per_epoch,
            alpha=0.01)

    model = get_unet(img_size=IMG_SIZE, in_channels=args.n_slices)
    _target = LOSSES[loss_name]
    if args.warmup_epochs > 0 and callable(_target):
        _target = make_warmup_loss(_target)
        print(f"Cross-entropy warm-up: {args.warmup_epochs} epochs of BCE, "
              f"then {loss_name}.")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss=_target,
        metrics=["accuracy", tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="io_u")],
    )
    model.summary()

    checkpoint = ModelCheckpoint(os.path.join(fold_dir, "model_patch_best.h5"),
                                 save_best_only=True, monitor="val_io_u",
                                 mode="max", verbose=1)
    csv_logger = CSVLogger(os.path.join(fold_dir, "training_log.csv"))

    print(f"\nTraining fold {args.fold} ({args.n_slices}-slice, recipe) for "
          f"{args.epochs} epochs, {steps_per_epoch} steps/epoch...")
    _cbs = [checkpoint, csv_logger]
    if args.warmup_epochs > 0 and callable(LOSSES[loss_name]):
        _cbs.append(WarmupSwitch(args.warmup_epochs))
    model.fit(train_seq, validation_data=val_seq, epochs=args.epochs,
              callbacks=_cbs,
              workers=1, use_multiprocessing=False)  # num_workers>0 hangs on Lustre

    print("\nLoading best checkpoint for inference...")
    model.load_weights(os.path.join(fold_dir, "model_patch_best.h5"))

    del X_train, Y_train, X_val, Y_val

    print(f"Running 3D sliding-window inference on {len(val_pids)} volumes...")
    dice_scores = run_inference_on_volumes(model, args.raw_dir, val_pids, args.n_slices)

    if dice_scores:
        avg_dice = float(np.mean(list(dice_scores.values())))
        std_dice = float(np.std(list(dice_scores.values())))
        print(f"\nFold {args.fold} ({args.n_slices}-slice, recipe) -- Mean Dice: "
              f"{avg_dice:.4f} +/- {std_dice:.4f}")
    else:
        avg_dice = std_dice = 0.0
        print("\nWarning: no Dice scores computed.")

    results = {
        "fold": args.fold,
        "seed": args.seed,
        "n_slices": args.n_slices,
        "recipe": recipe,
        "mean_dice": avg_dice,
        "std_dice": std_dice,
        "n_val_patients": len(val_pids),
        "per_volume_dice": dice_scores,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(fold_dir, "dice_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {os.path.join(fold_dir, 'dice_results.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
