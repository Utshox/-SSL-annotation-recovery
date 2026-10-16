"""
Mean Teacher and Cross-Pseudo Supervision, 5-fold, at UA-MT's exact configuration.
---------------------------------------------------------------------------------
The matched-control result so far rests on a single semi-supervised method, so
it cannot distinguish "UA-MT's reported gain was confounded" from "semi-supervised
gains in this setting are confounded generally". This script adds two more
methods under conditions identical to the UA-MT arm:

    same KFold(seed=42) patient split          same get_unet(dropout_rate=0.1)
    same seeded labeled/unlabeled halves        same BATCH_SIZE / EPOCHS / PATIENCE / LR
    same DualGenerator                          same volumetric inference and Dice

Because the configuration is identical, the supervised control already trained in
`baseline/models/supervised_frac_50` is the correct denominator for all three
methods. No new control is needed, which is what makes this affordable.

Mean Teacher is UA-MT without the uncertainty filtering, so it isolates that one
component. CPS is pseudo-labeling rather than consistency regularisation, so it
tests whether the confound extends beyond the consistency family.

Inference uses Gaussian patch blending by default, so these arms land directly on
the corrected footing and need no separate re-scoring pass.

Every improvement is checkpointed to disk, and `--inference_only` scores a
truncated run from its best checkpoint, so a walltime kill never costs the run.

Usage:
    python run_ssl_methods_5fold.py --method mt  --fold 0 --data_dir ... --raw_dir ...
    python run_ssl_methods_5fold.py --method cps --fold 0 --data_dir ... --raw_dir ...
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

from run_ssl_uamt import get_unet, DualGenerator, EpochCallback, load_and_concat
from run_ssl_meanteacher_v2 import MeanTeacherTrainer
from run_ssl_cps import CPSTrainer
# Config and inference come from the UA-MT arm so nothing can drift apart.
from run_ssl_uamt_5fold import (
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
    fold_split, get_patient_ids, run_inference_on_volumes, set_seed,
)

IMG_SIZE = 256


def make_average_model(model_a, model_b):
    """CPS predicts with the mean of its two networks, which is the standard
    protocol. Wrap them so the shared inference code sees a single model."""
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    out = layers.Average()([model_a(inp), model_b(inp)])
    return models.Model(inp, out)


class BestKeeper(tf.keras.callbacks.Callback):
    """Persist the evaluated network(s) to disk on every val_io_u improvement.

    UA-MT's keeper only handled a single teacher. CPS has two networks and needs
    both, since the prediction is their average.
    """

    def __init__(self, fold_dir, method):
        super().__init__()
        self.best = -1.0
        self.fold_dir = fold_dir
        self.method = method

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_io_u", -1.0)
        if v <= self.best:
            return
        self.best = v
        if self.method == "mt":
            self.model.teacher.save_weights(os.path.join(self.fold_dir, "teacher_best.h5"))
        else:
            self.model.model_a.save_weights(os.path.join(self.fold_dir, "model_a_best.h5"))
            self.model.model_b.save_weights(os.path.join(self.fold_dir, "model_b_best.h5"))
        with open(os.path.join(self.fold_dir, "best_info.json"), "w") as f:
            json.dump({"best_epoch": int(epoch), "best_val_io_u": float(v)}, f)


def build_eval_model(method, fold_dir):
    """Reconstruct the model used for scoring from whatever is on disk."""
    if method == "mt":
        ckpt = os.path.join(fold_dir, "teacher_best.h5")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        m = get_unet(dropout_rate=0.1)
        m.load_weights(ckpt)
        return m
    ck_a = os.path.join(fold_dir, "model_a_best.h5")
    ck_b = os.path.join(fold_dir, "model_b_best.h5")
    for c in (ck_a, ck_b):
        if not os.path.exists(c):
            raise FileNotFoundError(f"No checkpoint at {c}")
    ma, mb = get_unet(dropout_rate=0.1), get_unet(dropout_rate=0.1)
    ma.load_weights(ck_a)
    mb.load_weights(ck_b)
    return make_average_model(ma, mb)


def score_and_write(model, args, val_pids, fold_dir):
    dice = run_inference_on_volumes(model, args.raw_dir, val_pids, blend=args.blend)
    mean = float(np.mean(list(dice.values()))) if dice else 0.0
    std = float(np.std(list(dice.values()))) if dice else 0.0
    out = os.path.join(fold_dir, args.results_name)
    with open(out, "w") as f:
        json.dump({"method": args.method, "fold": args.fold, "blend": args.blend,
                   "mean_dice": mean, "std_dice": std,
                   "n": len(dice), "per_case": dice}, f, indent=2)
    print(f"\n=== {args.method} fold {args.fold}: mean 3D Dice = {mean:.4f} "
          f"+/- {std:.4f} (n={len(dice)}) -> {out} ===")


def main():
    ap = argparse.ArgumentParser(description="MT / CPS 5-fold at UA-MT's config.")
    ap.add_argument("--method", required=True, choices=["mt", "cps"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blend", default="gaussian", choices=["uniform", "gaussian"])
    ap.add_argument("--results_name", default="dice_results_gaussian.json")
    ap.add_argument("--inference_only", action="store_true")
    ap.add_argument("--frac", type=int, default=50, choices=[10, 25, 50],
                    help="Labeled share of the fold's training pool. Same seeded "
                         "permutation and formula as run_ssl_uamt_5fold and "
                         "run_supervised_fraction, so supervised_frac_<frac> is the "
                         "matched control. Default 50 reproduces the existing arms.")
    ap.add_argument("--train_seed", type=int, default=None,
                    help="Seed for weight init and sampling only; the split still uses --seed. Default: --seed.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--inner_val", type=int, default=0,
                    help="Corrected protocol: hold this many training-pool patients out "
                         "for early stopping/checkpoint selection (0 = legacy, outer fold).")
    ap.add_argument("--unlabeled_data_dir", default=None,
                    help="Label-free patch set for the unlabeled stream (corrected protocol).")
    args = ap.parse_args()

    out_root = args.output_dir or f"baseline/models/ssl_{args.method}_{args.frac}_5fold"
    set_seed(args.seed if args.train_seed is None else args.train_seed)
    fold_dir = os.path.join(out_root, f"fold_{args.fold}")
    Path(fold_dir).mkdir(parents=True, exist_ok=True)

    # Identical split and identical labeled draw to the UA-MT arm.
    patient_ids = get_patient_ids(args.data_dir)
    # Same helper as UA-MT and the supervised control, so all arms draw the same
    # patients. inner_val=0 reproduces the legacy split exactly.
    train_pids, labeled, unlabeled, sel_pids, val_pids = fold_split(
        patient_ids, args.fold, args.seed, args.frac, args.inner_val)
    print(f"Fold {args.fold}: train={len(train_pids)} "
          f"(labeled={len(labeled)}, unlabeled={len(unlabeled)}), selection={len(sel_pids)} "
          f"({'inner' if args.inner_val else 'OUTER fold'}), test={len(val_pids)}")

    with open(os.path.join(fold_dir, "split_info.json"), "w") as f:
        json.dump({"method": args.method, "fold": args.fold, "seed": args.seed,
                   "frac": args.frac,
                   "labeled": labeled, "unlabeled": unlabeled,
                   "validation": val_pids, "selection": sel_pids,
                   "inner_val": args.inner_val, "epochs": args.epochs,
                   "train_seed": args.train_seed,
                   "unlabeled_data_dir": args.unlabeled_data_dir or args.data_dir}, f, indent=2)

    if args.inference_only:
        score_and_write(build_eval_model(args.method, fold_dir), args, val_pids, fold_dir)
        return

    dd = args.data_dir
    X_l = load_and_concat([os.path.join(dd, f"{p}_x.npy") for p in labeled])
    Y_l = load_and_concat([os.path.join(dd, f"{p}_y.npy") for p in labeled])
    ud = args.unlabeled_data_dir or dd
    X_u = load_and_concat([os.path.join(ud, f"{p}_x.npy") for p in unlabeled])
    X_val = np.concatenate([np.load(os.path.join(dd, f"{p}_x.npy")) for p in sel_pids], axis=0)
    Y_val = np.concatenate([np.load(os.path.join(dd, f"{p}_y.npy")) for p in sel_pids], axis=0)
    X_val = np.expand_dims(X_val, -1).astype(np.float32)
    # Patch labels are {0,1,2}; the model is binary. DualGenerator clips the
    # training stream, so the validation stream must be clipped to match.
    Y_val = np.clip(np.expand_dims(Y_val, -1).astype(np.float32), 0, 1)

    train_gen = DualGenerator(X_l, Y_l, X_u, BATCH_SIZE)

    if args.method == "mt":
        student, teacher = get_unet(dropout_rate=0.1), get_unet(dropout_rate=0.1)
        teacher.set_weights(student.get_weights())
        model = MeanTeacherTrainer(student, teacher)
    else:
        model_a, model_b = get_unet(dropout_rate=0.1), get_unet(dropout_rate=0.1)
        model = CPSTrainer(model_a, model_b)

    model.compile(optimizer=optimizers.Adam(LR),
                  metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="io_u")])

    keeper = BestKeeper(fold_dir, args.method)
    callbacks = [keeper,
                 EarlyStopping(monitor="val_io_u", mode="max", patience=PATIENCE,
                               restore_best_weights=False, verbose=1),
                 CSVLogger(os.path.join(fold_dir, "log.csv")),
                 EpochCallback()]

    print(f"\nTraining {args.method.upper()} fold {args.fold} for up to {args.epochs} epochs "
          f"(batch {BATCH_SIZE}, LR {LR}, {len(train_gen)} steps/epoch)...")
    model.fit(train_gen, validation_data=(X_val, Y_val), epochs=args.epochs,
              callbacks=callbacks)
    print(f"Best val_io_u: {keeper.best:.4f}")

    del X_l, Y_l, X_u, X_val, Y_val
    score_and_write(build_eval_model(args.method, fold_dir), args, val_pids, fold_dir)


if __name__ == "__main__":
    main()
