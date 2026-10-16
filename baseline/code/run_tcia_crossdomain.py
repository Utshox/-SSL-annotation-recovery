"""
Cross-domain evaluation of the 5-fold arms on the external TCIA cohort.
-----------------------------------------------------------------------
The existing cross-domain numbers (supervised 0.544, UA-MT 0.603) came from the
older single-split models and were scored with uniform patch blending, so they
cannot be reported next to the five-fold, Gaussian-blended in-domain results.
This script rebuilds the cross-domain claim on the same models that produce the
in-domain one: each fold's checkpoint from `supervised_frac_50`,
`supervised_frac_100` or `ssl_uamt_50_5fold`, evaluated on all 80 TCIA volumes.

Every volume is an independent measurement, so results are streamed to a JSONL
file and flushed to disk as each one finishes, and the script skips volumes that
are already recorded when it restarts. A job that is cancelled or hits its
walltime at volume 60 leaves 60 usable results behind, and resubmitting costs
only the remaining 20. Two earlier runs in this project lost roughly 30 GPU-hours
by writing all of their output at the end; that failure mode is not possible here.

Usage:
    python run_tcia_crossdomain.py --arm uamt --fold 0 --blend gaussian
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from run_ssl_uamt import get_unet
from run_ssl_uamt_5fold import compute_dice, predict_sliding_window, set_seed
from run_ssl_methods_5fold import build_eval_model

HU_MIN, HU_MAX = -125, 275

# Single-checkpoint arms: the weights file is loaded straight into a U-Net.
ARMS = {
    "sup50":  ("baseline/models/supervised_frac_50",  "model_best.h5"),
    "sup100": ("baseline/models/supervised_frac_100", "model_best.h5"),
    "uamt":   ("baseline/models/ssl_uamt_50_5fold",   "teacher_best.h5"),
}

# MT and CPS are reconstructed by the same `build_eval_model` that scored them
# in-domain, so cross-domain uses the identical prediction function: the teacher
# for MT, and the average of the two networks for CPS. Duplicating the averaging
# here would risk the two evaluations silently diverging.
METHOD_ARMS = {
    "mt":  ("baseline/models/ssl_mt_50_5fold",  "mt"),
    "cps": ("baseline/models/ssl_cps_50_5fold", "cps"),
}

# Corrected-protocol arms (inner validation, label-free unlabeled pool). Runs
# trained on the second account live outside baseline/models: pass --model_root.
CLEAN_SINGLE = {f"c{k}{f}": (f"baseline/models/clean_{n}_{f}", ck)
                for k, n, ck in (("sup", "sup", "model_best.h5"),
                                 ("uamt", "uamt", "teacher_best.h5"),
                                 ("armb", "armb", "teacher_best.h5"))
                for f in (10, 25, 50, 100)}
CLEAN_METHOD = {f"c{m}{f}": (f"baseline/models/clean_{m}_{f}", m)
                for m in ("mt", "cps") for f in (10, 50)}
ARMS.update(CLEAN_SINGLE)
METHOD_ARMS.update(CLEAN_METHOD)


def load_done(path):
    """Volumes already scored in a previous (possibly killed) run."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["case"]] = rec["dice"]
            except (json.JSONDecodeError, KeyError):
                # A partial final line means the job died mid-write. Drop it and
                # let that one volume be recomputed.
                continue
    return done


def main():
    ap = argparse.ArgumentParser(description="Cross-domain TCIA evaluation.")
    ap.add_argument("--arm", required=True, choices=sorted(set(ARMS) | set(METHOD_ARMS)))
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--blend", default="gaussian", choices=["uniform", "gaussian"])
    ap.add_argument("--image_dir", default="data/tcia/imagesTs")
    ap.add_argument("--label_dir", default="data/tcia/labelsTs")
    ap.add_argument("--out_dir", default="baseline/logs/tcia_crossdomain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_root", default=None,
                    help="Override the arm's checkpoint directory (runs stored elsewhere).")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.arm in METHOD_ARMS:
        model_root, method = METHOD_ARMS[args.arm]
        model_root = args.model_root or model_root
        fold_dir = os.path.join(model_root, f"fold_{args.fold}")
        ckpt = fold_dir  # build_eval_model checks the individual files
    else:
        model_root, ckpt_name = ARMS[args.arm]
        model_root = args.model_root or model_root
        ckpt = os.path.join(model_root, f"fold_{args.fold}", ckpt_name)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"No checkpoint at {ckpt}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_fold{args.fold}_{args.blend}"
    jsonl = os.path.join(args.out_dir, f"partial_{tag}.jsonl")
    summary = os.path.join(args.out_dir, f"tcia_{tag}.json")

    images = sorted(Path(args.image_dir).glob("*.nii.gz"))
    done = load_done(jsonl)
    todo = [p for p in images if p.name.replace(".nii.gz", "") not in done]
    print(f"{tag}: {len(images)} volumes, {len(done)} already done, {len(todo)} to do.")
    if not todo:
        print("Nothing left to compute; writing summary from the partial file.")

    model = None
    if todo:
        if args.arm in METHOD_ARMS:
            model = build_eval_model(method, fold_dir)
        else:
            model = get_unet(dropout_rate=0.1)
            model.load_weights(ckpt)
        print(f"Loaded {ckpt} (blend={args.blend}).")

    # Append mode: a restart adds to what is already there.
    with open(jsonl, "a") as out:
        for img_path in tqdm(todo, desc=tag):
            case = img_path.name.replace(".nii.gz", "")
            lbl_path = Path(args.label_dir) / img_path.name
            if not lbl_path.exists():
                print(f"  Warning: no label for {case}, skipping.")
                continue

            vol_img = nib.load(str(img_path)).get_fdata().astype(np.float32)
            vol_lbl = nib.load(str(lbl_path)).get_fdata()

            vol_img = np.clip(vol_img, HU_MIN, HU_MAX)
            vol_img = (vol_img - HU_MIN) / (HU_MAX - HU_MIN)
            vol_img = np.clip(vol_img, 0.0, 1.0)

            pred_vol = np.zeros_like(vol_img)
            if vol_img.shape[0] == 512 and vol_img.shape[1] == 512:
                for s in range(vol_img.shape[2]):
                    pred_vol[:, :, s] = predict_sliding_window(
                        model, vol_img[:, :, s], blend=args.blend)
            else:
                for s in range(vol_img.shape[0]):
                    pred_vol[s, :, :] = predict_sliding_window(
                        model, vol_img[s, :, :], blend=args.blend)

            dice = float(compute_dice(vol_lbl, (pred_vol > 0.5).astype(np.float32)))

            # Persist immediately. flush() alone leaves the record in the OS
            # buffer, which a SLURM kill can discard, so fsync as well.
            out.write(json.dumps({"case": case, "dice": dice,
                                  "arm": args.arm, "fold": args.fold,
                                  "blend": args.blend}) + "\n")
            out.flush()
            os.fsync(out.fileno())
            print(f"  {case}: Dice = {dice:.4f}")

    scores = load_done(jsonl)
    mean = float(np.mean(list(scores.values()))) if scores else 0.0
    std = float(np.std(list(scores.values()))) if scores else 0.0
    with open(summary, "w") as f:
        json.dump({"arm": args.arm, "fold": args.fold, "blend": args.blend,
                   "checkpoint": ckpt, "n": len(scores),
                   "mean_dice": mean, "std_dice": std,
                   "per_case": scores, "complete": len(scores) >= len(images)},
                  f, indent=2)
    print(f"\n{tag}: mean Dice {mean:.4f} +/- {std:.4f} over {len(scores)}/{len(images)} volumes")
    if len(scores) < len(images):
        print("INCOMPLETE. Resubmit the same command to finish the remainder.")


if __name__ == "__main__":
    main()
