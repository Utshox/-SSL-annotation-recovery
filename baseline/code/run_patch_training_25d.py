"""
2.5D patch-based U-Net, 5-fold cross-validation
-----------------------------------------------
Tests whether the gap between the 2D patch CNN (binary 5-fold Dice 0.630) and
nnU-Net 3d_fullres (0.851) is explained by missing inter-slice context.

Identical to `run_patch_training_5fold.py` (same U-Net, same BCE loss, same
KFold(n_splits=5, shuffle=True, random_state=42) patient split, same 3D
sliding-window inference and volumetric Dice) except:

  * inputs come from `preprocessed_v6_25d` (N, 256, 256, 5) stacks and are
    sliced to the middle `--n_slices` channels;
  * the first Conv2D therefore sees `n_slices` input channels.

Run with --n_slices 1 for the matched 2D baseline on the identical patch set.
That controls for the patch re-extraction, so any difference at 3 or 5 slices
is attributable to inter-slice context alone.

Usage:
    python run_patch_training_25d.py --data_dir <v6 dir> --raw_dir <Task07> \
        --fold 0 --n_slices 3 --output_dir baseline/models/25d_3slice
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tqdm import tqdm

IMG_SIZE = 256
PATCH_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
CONTEXT_CHANNELS = 5  # what preprocess_v6_25d.py writes


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# U-Net, unchanged apart from the input channel count
# ---------------------------------------------------------------------------

def get_unet(img_size=256, in_channels=1, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, in_channels))

    c1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p4)
    c5 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c5)

    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(c9)
    return models.Model(inputs=[inputs], outputs=[outputs])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_patient_ids(data_dir):
    data_path = Path(data_dir)
    return sorted(set(f.name.replace("_x.npy", "") for f in data_path.glob("*_x.npy")))


def channel_slice(n_slices, total=CONTEXT_CHANNELS):
    """Middle `n_slices` channels of a `total`-channel stack (centre stays centred)."""
    if n_slices > total or n_slices % 2 == 0:
        raise ValueError(f"n_slices must be odd and <= {total}, got {n_slices}")
    half = n_slices // 2
    mid = total // 2
    return slice(mid - half, mid + half + 1)


class PatchSequence(tf.keras.utils.Sequence):
    """Feed batches from host RAM.

    Passing the full array to model.fit() makes Keras copy the whole dataset to
    the GPU, which OOMs a 32 GB V100 at 5 channels (20.5 GB of patches). Batch
    sizes and per-epoch shuffling match the array path, so every arm of the
    ablation trains through this same generator.
    """

    def __init__(self, X, Y, batch_size, shuffle, seed=42):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(X))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        return self.X[idx], self.Y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def load_patient_patches(data_dir, patient_ids, n_slices):
    """Load (N, 256, 256, n_slices) inputs and (N, 256, 256, 1) binary masks."""
    data_path = Path(data_dir)
    chans = channel_slice(n_slices)
    all_x, all_y = [], []

    for pid in patient_ids:
        xf, yf = data_path / f"{pid}_x.npy", data_path / f"{pid}_y.npy"
        if not xf.exists() or not yf.exists():
            print(f"  Warning: missing arrays for {pid}")
            continue
        bx = np.load(xf, mmap_mode="r")[..., chans]
        by = np.load(yf)
        if len(bx) > 0:
            all_x.append(np.ascontiguousarray(bx, dtype=np.float32))
            all_y.append(by)

    if not all_x:
        raise RuntimeError("No patches loaded -- check data_dir and patient IDs.")

    X = np.concatenate(all_x, axis=0)
    Y = np.expand_dims(np.concatenate(all_y, axis=0), axis=-1).astype(np.float32)
    Y = np.clip(Y, 0, 1)  # merge pancreas (1) and tumour (2) into one foreground
    return X, Y


# ---------------------------------------------------------------------------
# Sliding-window inference with neighbour-slice context
# ---------------------------------------------------------------------------

_GAUSSIAN_CACHE = {}


def gaussian_patch_weights(patch_size, sigma_scale=0.125):
    """Per-voxel patch weights that taper toward the patch border.

    Overlapping patches are averaged, and with uniform weights an interior
    voxel receives up to four votes while a border voxel receives one. Averaging
    confident 0/1 predictions across a differing number of votes drags the
    result toward the middle, which is exactly where the hardcoded 0.5 threshold
    sits, so the binarisation loses foreground that the model actually found.

    Weighting each patch by a Gaussian centred on the patch (sigma = 1/8 of the
    patch, the nnU-Net convention) makes each voxel's prediction come mostly
    from the patch that saw it with the most surrounding context. Costs no
    parameters and has nothing to tune.
    """
    key = (patch_size, sigma_scale)
    if key not in _GAUSSIAN_CACHE:
        sigma = patch_size * sigma_scale
        coords = np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0
        line = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        w = np.outer(line, line).astype(np.float32)
        w /= w.max()
        # Floor the tails so border voxels covered by only one patch still get a
        # usable vote instead of dividing by ~0.
        w = np.maximum(w, 1e-3)
        _GAUSSIAN_CACHE[key] = w
    return _GAUSSIAN_CACHE[key]


def predict_sliding_window_25d(model, vol, slice_idx, n_slices,
                               patch_size=PATCH_SIZE, stride=128,
                               blend="uniform"):
    """Predict one slice of a (D, H, W) volume using its neighbours as channels."""
    depth, h, w = vol.shape
    half = n_slices // 2
    idxs = [min(max(slice_idx + o, 0), depth - 1) for o in range(-half, half + 1)]
    stack = np.stack([vol[i] for i in idxs], axis=-1)  # (H, W, n_slices)

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    stack = np.pad(stack, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    h_pad, w_pad = stack.shape[0], stack.shape[1]

    prediction_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad), dtype=np.float32)

    patches, coords = [], []
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            patches.append(stack[y:y + patch_size, x:x + patch_size, :])
            coords.append((y, x))

    preds = model.predict(np.array(patches), batch_size=16, verbose=0)
    wmap = (gaussian_patch_weights(patch_size) if blend == "gaussian"
            else np.ones((patch_size, patch_size), dtype=np.float32))
    for i, (y, x) in enumerate(coords):
        prediction_map[y:y + patch_size, x:x + patch_size] += preds[i, :, :, 0] * wmap
        count_map[y:y + patch_size, x:x + patch_size] += wmap

    return (prediction_map / np.maximum(count_map, 1e-6))[:h, :w]


def compute_dice(y_true, y_pred):
    y_true = (y_true > 0).astype(np.float32)
    y_pred = (y_pred > 0).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 1.0
    return (2.0 * intersection) / denominator


def run_inference_on_volumes(model, raw_dir, patient_ids, n_slices,
                             blend="uniform", threshold=0.5):
    raw_path = Path(raw_dir)
    img_dir, lab_dir = raw_path / "imagesTr", raw_path / "labelsTr"
    HU_MIN, HU_MAX = -125, 275
    dice_scores = {}

    for pid in tqdm(patient_ids, desc="Inference"):
        img_path, lab_path = img_dir / f"{pid}.nii.gz", lab_dir / f"{pid}.nii.gz"
        if not img_path.exists() or not lab_path.exists():
            print(f"  Warning: NIfTI not found for {pid}, skipping.")
            continue

        vol_img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        vol_lbl = nib.load(str(lab_path)).get_fdata()

        vol_img = np.clip(vol_img, HU_MIN, HU_MAX)
        vol_img = (vol_img - HU_MIN) / (HU_MAX - HU_MIN)
        vol_img = np.clip(vol_img, 0.0, 1.0)

        # Work in (D, H, W) so the slice axis is axis 0, then map back.
        if vol_img.shape[0] == 512 and vol_img.shape[1] == 512:
            vol_dhw = np.transpose(vol_img, (2, 0, 1))
            pred_dhw = np.zeros_like(vol_dhw)
            for s in range(vol_dhw.shape[0]):
                pred_dhw[s] = predict_sliding_window_25d(model, vol_dhw, s, n_slices,
                                                        blend=blend)
            pred_vol = np.transpose(pred_dhw, (1, 2, 0))
        else:
            pred_vol = np.zeros_like(vol_img)
            for s in range(vol_img.shape[0]):
                pred_vol[s] = predict_sliding_window_25d(model, vol_img, s, n_slices,
                                                         blend=blend)

        dice = compute_dice(vol_lbl, (pred_vol > threshold).astype(np.float32))
        dice_scores[pid] = float(dice)
        print(f"  {pid}: Dice = {dice:.4f}")

    return dice_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2.5D patch U-Net, 5-fold CV.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="v6 2.5D patch directory")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--n_slices", type=int, default=3, choices=[1, 3, 5],
                        help="1 = matched 2D baseline; 3 or 5 = inter-slice context")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--blend", default="uniform", choices=["uniform", "gaussian"],
                        help="Patch-overlap weighting at inference. Default stays "
                             "uniform so every previously reported number remains "
                             "reproducible; gaussian is the corrected version.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--inference_only", action="store_true",
                        help="Re-score an existing model_patch_best.h5 without "
                             "retraining (used to re-run inference under --blend).")
    parser.add_argument("--results_name", default="dice_results.json",
                        help="Output filename, so a re-scored run does not "
                             "overwrite the original result.")
    args = parser.parse_args()

    set_seed(args.seed)
    fold_dir = os.path.join(args.output_dir, f"fold_{args.fold}")
    Path(fold_dir).mkdir(parents=True, exist_ok=True)

    patient_ids = get_patient_ids(args.data_dir)
    print(f"Found {len(patient_ids)} patients in {args.data_dir}")

    # Same split as run_patch_training_5fold.py -> directly comparable folds.
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = list(kf.split(patient_ids))[args.fold]
    train_pids = [patient_ids[i] for i in train_idx]
    val_pids = [patient_ids[i] for i in val_idx]
    print(f"\nFold {args.fold}: {len(train_pids)} train, {len(val_pids)} val patients")

    with open(os.path.join(fold_dir, "split_info.json"), "w") as f:
        json.dump({"fold": args.fold, "seed": args.seed, "n_slices": args.n_slices,
                   "n_train": len(train_pids), "n_val": len(val_pids),
                   "train_patients": train_pids, "val_patients": val_pids}, f, indent=2)

    if args.inference_only:
        ckpt = os.path.join(fold_dir, "model_patch_best.h5")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"--inference_only but no checkpoint at {ckpt}")
        model = get_unet(img_size=IMG_SIZE, in_channels=args.n_slices)
        model.load_weights(ckpt)
        print(f"Loaded {ckpt} for inference only "
              f"(blend={args.blend}, threshold={args.threshold}).")
        dice_scores = run_inference_on_volumes(model, args.raw_dir, val_pids,
                                               args.n_slices, blend=args.blend,
                                               threshold=args.threshold)
        avg = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
        std = float(np.std(list(dice_scores.values()))) if dice_scores else 0.0
        out = os.path.join(fold_dir, args.results_name)
        with open(out, "w") as f:
            json.dump({"fold": args.fold, "seed": args.seed,
                       "n_slices": args.n_slices, "blend": args.blend,
                       "threshold": args.threshold, "inference_only": True,
                       "mean_dice": avg, "std_dice": std,
                       "n_val_patients": len(val_pids),
                       "per_volume_dice": dice_scores,
                       "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=2)
        print(f"\nFold {args.fold} ({args.n_slices}-slice, blend={args.blend}) "
              f"-- Mean Dice: {avg:.4f} +/- {std:.4f}  -> {out}")
        return

    print(f"\nLoading patches ({args.n_slices}-slice input)...")
    X_train, Y_train = load_patient_patches(args.data_dir, train_pids, args.n_slices)
    X_val, Y_val = load_patient_patches(args.data_dir, val_pids, args.n_slices)
    print(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")

    model = get_unet(img_size=IMG_SIZE, in_channels=args.n_slices)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="io_u")],
    )
    model.summary()

    checkpoint = ModelCheckpoint(os.path.join(fold_dir, "model_patch_best.h5"),
                                 save_best_only=True, monitor="val_io_u",
                                 mode="max", verbose=1)
    csv_logger = CSVLogger(os.path.join(fold_dir, "training_log.csv"))

    train_seq = PatchSequence(X_train, Y_train, args.batch_size,
                              shuffle=True, seed=args.seed)
    val_seq = PatchSequence(X_val, Y_val, args.batch_size, shuffle=False)

    print(f"\nTraining fold {args.fold} ({args.n_slices}-slice) for {args.epochs} "
          f"epochs, {len(train_seq)} steps/epoch...")
    model.fit(train_seq, validation_data=val_seq, epochs=args.epochs,
              callbacks=[checkpoint, csv_logger],
              workers=1, use_multiprocessing=False)  # num_workers>0 hangs on Lustre

    print("\nLoading best checkpoint for inference...")
    best_path = os.path.join(fold_dir, "model_patch_best.h5")
    model.load_weights(best_path)

    # Free training arrays before volumetric inference.
    del X_train, Y_train, X_val, Y_val

    print(f"Running 3D sliding-window inference on {len(val_pids)} volumes...")
    dice_scores = run_inference_on_volumes(model, args.raw_dir, val_pids, args.n_slices)

    if dice_scores:
        avg_dice = float(np.mean(list(dice_scores.values())))
        std_dice = float(np.std(list(dice_scores.values())))
        print(f"\nFold {args.fold} ({args.n_slices}-slice) -- Mean Dice: "
              f"{avg_dice:.4f} +/- {std_dice:.4f}")
    else:
        avg_dice = std_dice = 0.0
        print("\nWarning: no Dice scores computed.")

    results = {
        "fold": args.fold,
        "seed": args.seed,
        "n_slices": args.n_slices,
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
