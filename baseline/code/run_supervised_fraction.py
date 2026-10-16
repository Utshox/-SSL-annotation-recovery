"""
Supervised control at a matched label fraction (the missing SSL denominator)
----------------------------------------------------------------------------
The headline SSL claim compares

    UA-MT   112 labeled + 112 unlabeled volumes  ->  0.651
    supervised   all 225 labeled volumes         ->  0.630

and concludes UA-MT "recovers 103% of full supervision". That comparison cannot
attribute the gain to the unlabeled data, because no supervised model was ever
trained on the labeled subset alone. It also confounds four other differences:
UA-MT uses a U-Net with dropout 0.1, batch size 8, 60 epochs with early
stopping, and a mean-teacher consistency term, while the supervised 5-fold uses
no dropout, batch size 32 and a fixed 100 epochs. Since the supervised baseline
overfits badly (fold-0 train IoU 0.981 vs val 0.614), any of those regularising
differences could produce the observed +0.021 on its own.

This script removes every one of those confounds. It is `run_ssl_uamt_5fold`
with the unlabeled stream and the consistency loss deleted, and nothing else
changed: same KFold(seed=42) patient split, same deterministic labeled subset,
same `get_unet(dropout_rate=0.1)`, same batch size, same epoch budget and early
stopping, same validation patches, same sliding-window inference, same Dice and
the same `per_case` output key. So

    UA-MT(frac) - supervised(frac)

is attributable to the unlabeled data alone, and running frac in {10, 25, 50}
gives a label-efficiency curve with a proper denominator.

Usage:
    python run_supervised_fraction.py --data_dir <v5 patches> --raw_dir <Task07> \
        --fold 0 --frac 50 --output_dir baseline/models/supervised_frac_50
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

# Same network and loader as the SSL arm, so the only difference is the data.
from run_ssl_uamt import get_unet, load_and_concat
from run_ssl_uamt_5fold import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    PATIENCE,
    augment_batch,
    fold_split,
    get_patient_ids,
    run_inference_on_volumes,
    set_seed,
)


def main():
    ap = argparse.ArgumentParser(
        description="Supervised control at a matched label fraction.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--frac", type=int, required=True, choices=[10, 25, 50, 100],
                    help="Percent of each fold's training pool used as labeled")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_seed", type=int, default=None,
                    help="Seed for weight init and sampling only; the split still uses --seed. Default: --seed.")
    ap.add_argument("--inference_only", action="store_true")
    ap.add_argument("--blend", default="uniform", choices=["uniform", "gaussian"],
                    help="Patch-overlap weighting at inference; see run_ssl_uamt_5fold.")
    ap.add_argument("--results_name", default="dice_results.json")
    ap.add_argument("--aug_style", default="generic", choices=["generic", "abayomi"])
    ap.add_argument("--es_start", type=int, default=0,
                    help="EarlyStopping start_from_epoch. Augmented arms sit at val IoU 0 "
                         "for 12-16 epochs, so patience 15 can stop them before they learn.")
    ap.add_argument("--aug", action="store_true",
                    help="Same augment_batch as the SSL arm (--aug there).")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="Epoch cap; raise together with the SSL arm so the pair stays matched.")
    ap.add_argument("--inner_val", type=int, default=0,
                    help="See run_ssl_uamt_5fold. 0 reproduces the legacy protocol.")
    args = ap.parse_args()

    set_seed(args.seed if args.train_seed is None else args.train_seed)
    fold_dir = os.path.join(args.output_dir, f"fold_{args.fold}")
    Path(fold_dir).mkdir(parents=True, exist_ok=True)

    # Same helper as the SSL arm, so the labeled set (and any inner selection
    # split) is the SAME set of patients the SSL arm uses.
    patient_ids = get_patient_ids(args.data_dir)
    train_pids, labeled, _, sel_pids, val_pids = fold_split(
        patient_ids, args.fold, args.seed, args.frac, args.inner_val)
    if args.frac == 100:
        labeled = train_pids
    print(f"Fold {args.fold}: train pool={len(train_pids)}, labeled={len(labeled)} "
          f"({args.frac}%), selection={len(sel_pids)} "
          f"({'inner' if args.inner_val else 'OUTER fold'}), test={len(val_pids)}  [no unlabeled stream]")

    with open(os.path.join(fold_dir, "split_info.json"), "w") as f:
        json.dump({"fold": args.fold, "frac": args.frac, "seed": args.seed,
                   "labeled": labeled, "validation": val_pids,
                   "selection": sel_pids, "inner_val": args.inner_val}, f, indent=2)

    ckpt = os.path.join(fold_dir, "model_best.h5")

    if args.inference_only:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"--inference_only but no checkpoint at {ckpt}")
        model = get_unet(dropout_rate=0.1)
        model.load_weights(ckpt)
        dice_scores = run_inference_on_volumes(model, args.raw_dir, val_pids,
                                               blend=args.blend)
        mean_dice = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
        with open(os.path.join(fold_dir, args.results_name), "w") as f:
            json.dump({"fold": args.fold, "frac": args.frac, "mean_dice": mean_dice,
                       "blend": args.blend,
                       "n": len(dice_scores), "per_case": dice_scores}, f, indent=2)
        print(f"=== Fold {args.fold} supervised-{args.frac}% (inference-only): "
              f"{mean_dice:.4f} (n={len(dice_scores)}) ===")
        return

    dd = args.data_dir
    X_l = load_and_concat([os.path.join(dd, f"{p}_x.npy") for p in labeled])
    Y_l = load_and_concat([os.path.join(dd, f"{p}_y.npy") for p in labeled])
    # The SSL arm applies this inside DualGenerator.__getitem__; here the arrays
    # go straight to fit(), so it has to happen up front. Patch labels are
    # {0, 1, 2} (2 = tumor) and the model is binary, so the clip is required:
    # without it BinaryIoU's confusion matrix aborts on an out-of-bound label.
    X_l = np.expand_dims(X_l, -1).astype(np.float32)
    Y_l = np.clip(np.expand_dims(Y_l, -1).astype(np.float32), 0, 1)
    X_val = np.concatenate([np.load(os.path.join(dd, f"{p}_x.npy")) for p in sel_pids], axis=0)
    Y_val = np.concatenate([np.load(os.path.join(dd, f"{p}_y.npy")) for p in sel_pids], axis=0)
    X_val = np.expand_dims(X_val, -1).astype(np.float32)
    Y_val = np.clip(np.expand_dims(Y_val, -1).astype(np.float32), 0, 1)
    print(f"  Labeled patches: X={X_l.shape}  Val patches: X={X_val.shape}")

    model = get_unet(dropout_rate=0.1)
    model.compile(optimizer=optimizers.Adam(LR), loss="binary_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="io_u")])

    # Checkpoint to disk on every improvement, mirroring BestTeacherKeeper, so a
    # walltime timeout still leaves a usable model for --inference_only.
    callbacks = [
        ModelCheckpoint(ckpt, save_best_only=True, save_weights_only=True,
                        monitor="val_io_u", mode="max", verbose=1),
        EarlyStopping(monitor="val_io_u", mode="max", patience=PATIENCE,
                          start_from_epoch=args.es_start,
                      restore_best_weights=False, verbose=1),
        CSVLogger(os.path.join(fold_dir, "log.csv")),
    ]

    print(f"\nTraining supervised-{args.frac}% fold {args.fold} for up to "
          f"{args.epochs} epochs (batch {BATCH_SIZE}, LR {LR})...")
    if args.aug:
        import run_ssl_uamt_5fold as _u
        _u.AUG_STYLE = args.aug_style
        # Same steps/epoch and batch size as the array fit below; only the
        # per-batch augmentation is added.
        class AugSeq(tf.keras.utils.Sequence):
            def __init__(self):
                self.perm = np.random.permutation(len(X_l))
            def __len__(self):
                return int(np.ceil(len(X_l) / BATCH_SIZE))
            def __getitem__(self, i):
                ix = self.perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                return augment_batch(X_l[ix], Y_l[ix])
            def on_epoch_end(self):
                self.perm = np.random.permutation(len(X_l))
        model.fit(AugSeq(), validation_data=(X_val, Y_val), epochs=args.epochs,
                  callbacks=callbacks)
    else:
        model.fit(X_l, Y_l, validation_data=(X_val, Y_val), epochs=args.epochs,
                  batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks)

    if os.path.exists(ckpt):
        model.load_weights(ckpt)

    # Used to drop blend= and hardcode dice_results.json (same bug fixed in the SSL
    # script on 2026-09-03); defaults unchanged so old arms reproduce.
    dice_scores = run_inference_on_volumes(model, args.raw_dir, val_pids, blend=args.blend)
    mean_dice = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
    with open(os.path.join(fold_dir, args.results_name), "w") as f:
        json.dump({"fold": args.fold, "frac": args.frac, "mean_dice": mean_dice,
                   "blend": args.blend,
                   "n": len(dice_scores), "per_case": dice_scores}, f, indent=2)
    print(f"=== Fold {args.fold} supervised-{args.frac}%: mean 3D Dice = "
          f"{mean_dice:.4f} (n={len(dice_scores)}) ===")


if __name__ == "__main__":
    main()
