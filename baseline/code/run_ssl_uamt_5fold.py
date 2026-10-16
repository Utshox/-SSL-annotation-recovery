"""
5-Fold UA-MT (50% labeled) for equal-footing SSL evaluation.

Mirrors run_patch_training_5fold.py's patient-level KFold(seed=42) splits and
3D sliding-window inference, but trains the Uncertainty-Aware Mean Teacher
(student/teacher EMA) using 50% of each fold's training pool as labeled and the
remainder as unlabeled. After training, runs 3D inference on the held-out fold's
raw NIfTI volumes with the EMA teacher and reports per-volume 3D Dice.

This produces a population-level UA-MT-50% Dice over all 281 volumes directly
comparable to the supervised 5-fold number (0.632).

Usage:
    python run_ssl_uamt_5fold.py --data_dir <patches> --raw_dir <Task07> \
        --fold N --output_dir baseline/models/ssl_uamt_50_5fold --seed 42
"""
import os
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import KFold
from tqdm import tqdm

# Reuse the exact UA-MT trainer + unet from the proven single-split script
from run_ssl_uamt import get_unet, UAMTTrainer, DualGenerator, EpochCallback, load_and_concat
# One definition of the patch-blending kernel, shared with the 2.5D pipeline.
from run_patch_training_25d import gaussian_patch_weights


AUG_STYLE = "generic"   # set from --aug_style in both scripts


def _augment_abayomi(x, y):
    """Abayomi-Alli et al. 2026 (ICHCAI) lung pipeline: rotation +-10 deg, horizontal
    flip, zoom +-10%, intensity shift. Probabilities are not reported there; rotation,
    zoom and shift are always drawn, flip with p=0.5. One affine per sample, bilinear
    for the image and nearest for the label so the mask stays binary."""
    import cv2
    x = x.copy()
    y = None if y is None else y.copy()
    h, w = x.shape[1:3]
    for i in range(len(x)):
        m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0),
                                    np.random.uniform(-10, 10), np.random.uniform(0.9, 1.1))
        xi = cv2.warpAffine(x[i, ..., 0], m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fh = np.random.rand() < 0.5
        if fh: xi = xi[:, ::-1]
        x[i, ..., 0] = np.clip(xi + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        if y is not None:
            yi = cv2.warpAffine(y[i, ..., 0], m, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if fh: yi = yi[:, ::-1]
            y[i, ..., 0] = yi
    return x, y


def augment_batch(x, y=None):
    """Per-sample augmentation shared by the supervised and SSL arms (--aug).

    Random horizontal/vertical flip, random 90-degree rotation, and an intensity
    scale/shift clipped back to [0, 1]. The geometric ops hit image and label alike;
    intensity only touches the image. Both scripts call this one function so the
    augmentation distribution is identical across arms. x, y: [B, H, W, 1].
    --aug_style abayomi swaps in _augment_abayomi.
    """
    if AUG_STYLE == "abayomi":
        return _augment_abayomi(x, y)
    x = x.copy()
    y = None if y is None else y.copy()
    for i in range(len(x)):
        k = np.random.randint(4)
        fh, fv = np.random.rand() < 0.5, np.random.rand() < 0.5
        xi = np.rot90(x[i], k, axes=(0, 1))
        if fh: xi = xi[:, ::-1]
        if fv: xi = xi[::-1, :]
        xi = xi * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        x[i] = np.clip(xi, 0.0, 1.0)
        if y is not None:
            yi = np.rot90(y[i], k, axes=(0, 1))
            if fh: yi = yi[:, ::-1]
            if fv: yi = yi[::-1, :]
            y[i] = yi
    return x, y


class AugDualGenerator(DualGenerator):
    """DualGenerator with augment_batch applied to both the labeled and the
    consistency half. Teacher and student see the same augmented x_u."""
    def __getitem__(self, idx):
        (xl, yl), xu = super().__getitem__(idx)
        xl, yl = augment_batch(xl, yl)
        xu, _ = augment_batch(xu)
        return (xl, yl), xu

BATCH_SIZE = 8
EPOCHS = 60          # converges by ~ep40 (was 100, wasteful); early-stops below
PATIENCE = 15        # early stop on val_io_u plateau
LR = 1e-4
T = 8
RATIO = 50  # percent labeled within each fold's training pool


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_patient_ids(data_dir):
    data_path = Path(data_dir)
    x_files = sorted(data_path.glob("*_x.npy"))
    return sorted(set(f.name.replace("_x.npy", "") for f in x_files))


# --- 3D inference (copied verbatim from run_patch_training_5fold.py) ---
def predict_sliding_window(model, image, patch_size=256, stride=128, blend="uniform"):
    """Reconstruct a full slice from overlapping patch predictions.

    `blend` mirrors run_patch_training_25d: "uniform" reproduces every number
    published before 2026-08-03, "gaussian" removes the unequal-vote artifact
    that drags reconstructed probabilities toward the 0.5 threshold.
    """
    h, w = image.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
    h_pad, w_pad = image_padded.shape
    prediction_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    patches, coords = [], []
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            patches.append(image_padded[y:y + patch_size, x:x + patch_size])
            coords.append((y, x))
    patches = np.expand_dims(np.array(patches), axis=-1)
    preds = model.predict(patches, batch_size=16, verbose=0)
    wmap = (gaussian_patch_weights(patch_size) if blend == "gaussian"
            else np.ones((patch_size, patch_size), dtype=np.float32))
    for i, (y, x) in enumerate(coords):
        prediction_map[y:y + patch_size, x:x + patch_size] += preds[i, :, :, 0] * wmap
        count_map[y:y + patch_size, x:x + patch_size] += wmap
    avg_pred = prediction_map / np.maximum(count_map, 1e-6)
    return avg_pred[:h, :w]


def compute_dice(y_true, y_pred):
    y_true = (y_true > 0).astype(np.float32)
    y_pred = (y_pred > 0).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 1.0
    return (2.0 * intersection) / denominator


def run_inference_on_volumes(model, raw_dir, patient_ids, blend="uniform"):
    raw_path = Path(raw_dir)
    img_dir = raw_path / "imagesTr"
    lab_dir = raw_path / "labelsTr"
    HU_MIN, HU_MAX = -125, 275
    dice_scores = {}
    for pid in tqdm(patient_ids, desc="Inference"):
        img_path = img_dir / f"{pid}.nii.gz"
        lab_path = lab_dir / f"{pid}.nii.gz"
        if not img_path.exists() or not lab_path.exists():
            print(f"  Warning: NIfTI not found for {pid}, skipping.")
            continue
        vol_img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        vol_lbl = nib.load(str(lab_path)).get_fdata()
        vol_img = np.clip(vol_img, HU_MIN, HU_MAX)
        vol_img = (vol_img - HU_MIN) / (HU_MAX - HU_MIN)
        vol_img = np.clip(vol_img, 0.0, 1.0)
        if vol_img.shape[0] == 512 and vol_img.shape[1] == 512:
            pred_vol = np.zeros_like(vol_img)
            for s in range(vol_img.shape[2]):
                pred_vol[:, :, s] = predict_sliding_window(model, vol_img[:, :, s], blend=blend)
        else:
            pred_vol = np.zeros_like(vol_img)
            for s in range(vol_img.shape[0]):
                pred_vol[s, :, :] = predict_sliding_window(model, vol_img[s, :, :], blend=blend)
        pred_bin = (pred_vol > 0.5).astype(np.float32)
        dice = compute_dice(vol_lbl, pred_bin)
        dice_scores[pid] = float(dice)
        print(f"  {pid}: Dice = {dice:.4f}")
    return dice_scores


class BestTeacherKeeper(tf.keras.callbacks.Callback):
    """Save the EMA teacher's weights TO DISK whenever val_io_u improves, so a
    walltime timeout still leaves the best checkpoint recoverable (the old version
    only kept weights in memory and wrote them after fit() -> lost on timeout)."""
    def __init__(self, fold_dir):
        super().__init__()
        self.best = -1.0
        self.fold_dir = fold_dir
        self.ckpt = os.path.join(fold_dir, "teacher_best.h5")

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_io_u", -1.0)
        if v > self.best:
            self.best = v
            self.model.teacher.save_weights(self.ckpt)
            with open(os.path.join(self.fold_dir, "best_info.json"), "w") as f:
                json.dump({"best_epoch": int(epoch), "best_val_io_u": float(v)}, f)


def fold_split(patient_ids, fold, seed, frac, inner_val=0):
    """Outer fold, optional inner selection split, and the seeded labeled draw.

    Returns (pool, labeled, unlabeled, select_pids, test_pids). test_pids is the
    outer fold and is only ever scored. select_pids drives early stopping and
    checkpoint selection: with inner_val=0 it IS the outer fold (the protocol every
    arm before 2026-09-28 used, reproduced exactly); with inner_val>0 it is that
    many patients held out of the training pool, so the scored fold never
    influences training. Shared by the SSL and supervised scripts so the two arms
    cannot draw different patients.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = list(kf.split(patient_ids))[fold]
    pool = [patient_ids[i] for i in train_idx]
    test_pids = [patient_ids[i] for i in val_idx]
    select_pids = test_pids
    if inner_val:
        # Separate stream from the labeled permutation below, so the inner split
        # does not depend on, or perturb, how labeled patients are drawn.
        iperm = np.random.RandomState(seed + 1000).permutation(len(pool))
        select_pids = [pool[i] for i in iperm[:inner_val]]
        pool = [pool[i] for i in iperm[inner_val:]]
    perm = np.random.RandomState(seed).permutation(len(pool))
    n_lab = len(pool) // 2 if frac == 50 else int(len(pool) * frac / 100)
    labeled = [pool[i] for i in perm[:n_lab]]
    unlabeled = [pool[i] for i in perm[n_lab:]]
    return pool, labeled, unlabeled, select_pids, test_pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--output_dir", default=None,
                    help="Defaults to baseline/models/ssl_uamt_<frac>_5fold. Derived from "
                         "--frac rather than fixed, so a fraction run can never land in "
                         "(and overwrite) the finished 50%% arm.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_seed", type=int, default=None,
                    help="Seed for weight init and sampling only; the split still uses --seed. Default: --seed.")
    ap.add_argument("--consistency_source", choices=["unlabeled", "labeled"],
                    default="unlabeled",
                    help="Which patient pool feeds the consistency branch. "
                         "'unlabeled' is UA-MT proper (arm C): the branch sees the "
                         "113 patients whose labels are never read. 'labeled' is the "
                         "arm-B control: the branch replays the SAME labeled patients, "
                         "with their labels still withheld from it, so the consistency "
                         "mechanism runs on images already represented in supervised "
                         "training. C minus B is then the incremental effect of distinct "
                         "unannotated patients under an otherwise identical procedure.")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="Epoch cap. An epoch is one pass over the LABELED patches, so at 10%% "
                         "labels it is ~5x fewer optimizer steps than at 50%%, and UA-MT@10 hit "
                         "the default cap of 60 in all 5 folds (best epoch 58-59) while its "
                         "matched control early-stopped. Raise it so the cap is non-binding.")
    ap.add_argument("--inner_val", type=int, default=0,
                    help="Hold this many patients out of the training pool for early stopping "
                         "and checkpoint selection. 0 (default) reproduces the legacy protocol, "
                         "which selected on the scored outer fold.")
    ap.add_argument("--unlabeled_data_dir", default=None,
                    help="Patch directory for the unlabeled stream. v5 patches were centred on "
                         "ground-truth pancreas for EVERY patient; point this at the label-free "
                         "v7 set so the unlabeled stream uses no annotation.")
    ap.add_argument("--frac", type=int, default=50, choices=[10, 25, 50, 100],
                    help="Percent of the fold's training pool used as labeled. The "
                         "remainder becomes the unlabeled stream. Draws from the same "
                         "seeded permutation as run_supervised_fraction, so the labeled "
                         "sets are nested and match the supervised control exactly.")
    ap.add_argument("--inference_only", action="store_true",
                    help="Skip training; load teacher_best.h5 and run inference + write dice_results.json. "
                         "Use to finish a fold whose training timed out after saving its best checkpoint.")
    ap.add_argument("--blend", default="uniform", choices=["uniform", "gaussian"],
                    help="Patch-overlap weighting at inference. Default stays uniform so "
                         "previously reported numbers remain reproducible.")
    ap.add_argument("--aug_style", default="generic", choices=["generic", "abayomi"],
                    help="generic = rot90/flips/intensity; abayomi = her 2026 lung pipeline.")
    ap.add_argument("--es_start", type=int, default=0,
                    help="EarlyStopping start_from_epoch. Augmented arms sit at val IoU 0 "
                         "for 12-16 epochs, so patience 15 can stop them before they learn.")
    ap.add_argument("--aug", action="store_true",
                    help="Apply augment_batch (flip/rot90/intensity) to labeled and "
                         "consistency batches. Off by default so old arms reproduce.")
    ap.add_argument("--results_name", default="dice_results.json",
                    help="Output filename, so a re-scored run does not overwrite the original.")
    args = ap.parse_args()

    set_seed(args.seed if args.train_seed is None else args.train_seed)
    out_root = args.output_dir or f"baseline/models/ssl_uamt_{args.frac}_5fold"
    fold_dir = os.path.join(out_root, f"fold_{args.fold}")
    Path(fold_dir).mkdir(parents=True, exist_ok=True)

    # Identical KFold(42) splits as the supervised 5-fold; see fold_split.
    patient_ids = get_patient_ids(args.data_dir)
    train_pids, labeled, unlabeled, sel_pids, val_pids = fold_split(
        patient_ids, args.fold, args.seed, args.frac, args.inner_val)
    print(f"Fold {args.fold}: train={len(train_pids)} (labeled={len(labeled)}, "
          f"unlabeled={len(unlabeled)}), selection={len(sel_pids)} "
          f"({'inner' if args.inner_val else 'OUTER fold'}), test={len(val_pids)}")

    with open(os.path.join(fold_dir, "split_info.json"), "w") as f:
        json.dump({"labeled": labeled, "unlabeled": unlabeled, "validation": val_pids,
                   "selection": sel_pids, "inner_val": args.inner_val,
                   "unlabeled_data_dir": args.unlabeled_data_dir or args.data_dir, "aug": args.aug,
                   "consistency_source": args.consistency_source}, f, indent=2)

    ckpt = os.path.join(fold_dir, "teacher_best.h5")

    # --- inference-only: load best teacher checkpoint, run 3D inference, exit ---
    if args.inference_only:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"--inference_only but no checkpoint at {ckpt}")
        teacher = get_unet(dropout_rate=0.1)
        teacher.load_weights(ckpt)
        print(f"Loaded {ckpt} for inference-only.")
        dice_scores = run_inference_on_volumes(teacher, args.raw_dir, val_pids,
                                               blend=args.blend)
        mean_dice = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
        with open(os.path.join(fold_dir, args.results_name), "w") as f:
            json.dump({"fold": args.fold, "mean_dice": mean_dice, "blend": args.blend,
                       "n": len(dice_scores), "per_case": dice_scores}, f, indent=2)
        print(f"=== Fold {args.fold} UA-MT-50% (inference-only): mean 3D Dice = "
              f"{mean_dice:.4f} (n={len(dice_scores)}) ===")
        return

    dd = args.data_dir
    X_l = load_and_concat([os.path.join(dd, f"{p}_x.npy") for p in labeled])
    Y_l = load_and_concat([os.path.join(dd, f"{p}_y.npy") for p in labeled])
    if args.consistency_source == "labeled":
        # Arm B. The consistency branch replays the labeled patients' images. Y_l is
        # never handed to it, so their annotations stay out of the consistency term
        # exactly as the unlabeled patients' would. DualGenerator draws from x_u with
        # its own np.random.choice, so the consistency half of a batch is sampled
        # independently of the supervised half rather than duplicating it.
        X_u = X_l
        cons_pids = labeled
    else:
        ud = args.unlabeled_data_dir or dd
        X_u = load_and_concat([os.path.join(ud, f"{p}_x.npy") for p in unlabeled])
        assert len(X_u) > 0, f"no unlabeled patches found under {ud}"
        cons_pids = unlabeled
    # Selection patches: the outer fold by default (legacy), inner split with --inner_val.
    X_val = np.concatenate([np.load(os.path.join(dd, f"{p}_x.npy")) for p in sel_pids], axis=0)
    Y_val = np.concatenate([np.load(os.path.join(dd, f"{p}_y.npy")) for p in sel_pids], axis=0)
    X_val = np.expand_dims(X_val, -1).astype(np.float32)
    Y_val = np.clip(np.expand_dims(Y_val, -1).astype(np.float32), 0, 1)

    global AUG_STYLE
    AUG_STYLE = args.aug_style
    train_gen = (AugDualGenerator if args.aug else DualGenerator)(X_l, Y_l, X_u, BATCH_SIZE)
    # Everything the two arms must share is a function of X_l alone: steps per epoch
    # is len(X_l)//(BATCH_SIZE//2), and each step draws BATCH_SIZE//2 consistency
    # images whatever the size of X_u. So optimizer steps, batch composition and the
    # number of consistency examples per epoch are identical across arms B and C, and
    # the pool the consistency images come from is the only thing that differs.
    print(f"consistency_source={args.consistency_source}: "
          f"{len(cons_pids)} patients, {len(X_u)} patches feed the consistency branch")
    print(f"steps/epoch={len(train_gen)}  supervised patches={len(X_l)}  "
          f"consistency examples/epoch={len(train_gen) * (BATCH_SIZE // 2)}")
    if args.consistency_source == "labeled":
        assert set(cons_pids) == set(labeled), "arm B must replay exactly the labeled patients"
        assert len(X_u) == len(X_l), "arm B consistency pool must be the labeled patches"

    student = get_unet(dropout_rate=0.1)
    teacher = get_unet(dropout_rate=0.1)
    teacher.set_weights(student.get_weights())

    model = UAMTTrainer(student, teacher, T=T)
    model.compile(optimizer=optimizers.Adam(LR),
                  metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="io_u")])

    keeper = BestTeacherKeeper(fold_dir)
    early = EarlyStopping(monitor="val_io_u", mode="max", patience=PATIENCE,
                          start_from_epoch=args.es_start,
                          restore_best_weights=False, verbose=1)
    model.fit(train_gen, validation_data=(X_val, Y_val), epochs=args.epochs,
              callbacks=[keeper, early, CSVLogger(os.path.join(fold_dir, "log.csv")), EpochCallback()])

    # Best EMA teacher was already saved to disk by the keeper each improvement.
    # Reload it (the live teacher may be past its best) before inference.
    if os.path.exists(ckpt):
        teacher.load_weights(ckpt)
    print(f"Best val IoU: {keeper.best:.4f}")

    # This path used to drop blend= and so always scored with uniform blending,
    # silently ignoring --blend and writing to the fixed dice_results.json. Arms
    # trained here therefore needed a separate Gaussian rescore pass to be
    # comparable with the rest of the paper. Both flags are honoured now.
    dice_scores = run_inference_on_volumes(teacher, args.raw_dir, val_pids,
                                           blend=args.blend)
    mean_dice = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
    with open(os.path.join(fold_dir, args.results_name), "w") as f:
        json.dump({"fold": args.fold, "mean_dice": mean_dice, "blend": args.blend,
                   "n": len(dice_scores), "per_case": dice_scores}, f, indent=2)
    print(f"=== Fold {args.fold} UA-MT-50%: mean 3D Dice = {mean_dice:.4f} (n={len(dice_scores)}) ===")


if __name__ == "__main__":
    main()
