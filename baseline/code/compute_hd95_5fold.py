#!/usr/bin/env python
"""HD95 and ASD for the five-fold arms, scored exactly as Dice is.

The existing HD95 numbers in `baseline/logs/hd95/` are unusable: they were
computed on the four-volume held-out split whose patches had entered training
(their Dice of 0.803 is the leaked figure), and on the older standalone
checkpoints rather than the five-fold ones. This script recomputes distance
metrics on the same out-of-fold predictions that produce the Dice spine, using
the same Gaussian-blended sliding-window inference, so HD95 and Dice describe
the identical segmentations.

Distance metrics are computed in millimetres using each volume's own voxel
spacing from the NIfTI header, not in voxel units, since slice thickness varies
across this cohort.

Empty predictions have no defined surface distance. Rather than silently
dropping those volumes, which would bias HD95 downward by removing the worst
cases, the count is reported and they are excluded only from the distance
summary, never from the Dice summary.

Per-volume results are streamed to JSONL and completed volumes are skipped on
restart, so a job killed by the walltime or the GPU quota keeps its work.

Usage:
    python baseline/code/compute_hd95_5fold.py --arm sup50 --fold 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_hd95 import compute_hd95, compute_asd, compute_dice
from run_ssl_uamt import get_unet
from run_ssl_uamt_5fold import predict_sliding_window, set_seed

HU_MIN, HU_MAX = -125, 275

SINGLE = {
    "sup50":  ("baseline/models/supervised_frac_50",  "model_best.h5"),
    "sup100": ("baseline/models/supervised_frac_100", "model_best.h5"),
    "uamt":   ("baseline/models/ssl_uamt_50_5fold",   "teacher_best.h5"),
    "sup10":  ("baseline/models/supervised_frac_10",  "model_best.h5"),
    "sup25":  ("baseline/models/supervised_frac_25",  "model_best.h5"),
    "uamt10": ("baseline/models/ssl_uamt_10_5fold",   "teacher_best.h5"),
    "uamt25": ("baseline/models/ssl_uamt_25_5fold",   "teacher_best.h5"),
    # Arm B: UA-MT with the consistency pool drawn from the labeled patients.
    "armb10": ("baseline/models/ssl_armb_10_5fold",   "teacher_best.h5"),
    "armb25": ("baseline/models/ssl_armb_25_5fold",   "teacher_best.h5"),
    "armb50": ("baseline/models/ssl_armb_50_5fold",   "teacher_best.h5"),
}
METHOD = {
    "mt":  ("baseline/models/ssl_mt_50_5fold",  "mt"),
    "cps": ("baseline/models/ssl_cps_50_5fold", "cps"),
    "mt10":  ("baseline/models/ssl_mt_10_5fold",  "mt"),
    "mt25":  ("baseline/models/ssl_mt_25_5fold",  "mt"),
    "cps10": ("baseline/models/ssl_cps_10_5fold", "cps"),
    "cps25": ("baseline/models/ssl_cps_25_5fold", "cps"),
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
SINGLE.update(CLEAN_SINGLE)
METHOD.update(CLEAN_METHOD)


def load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["case"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def val_pids(model_root, fold):
    """The held-out patients for this fold, read from the run's own split file."""
    with open(os.path.join(model_root, f"fold_{fold}", "split_info.json")) as f:
        info = json.load(f)
    for key in ("validation", "val_pids", "val_patients", "val", "test_pids"):
        if key in info:
            return list(info[key])
    raise KeyError(f"no validation patient list in split_info.json: {list(info)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(set(SINGLE) | set(METHOD)))
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--raw_dir", default="data/Task07_Pancreas")
    ap.add_argument("--out_dir", default="baseline/logs/hd95_5fold")
    ap.add_argument("--blend", default="gaussian", choices=["uniform", "gaussian"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_root", default=None,
                    help="Override the arm's checkpoint directory (runs stored elsewhere).")
    ap.add_argument("--also_lcc", action="store_true",
                    help="Additionally score the largest 3D connected component of the "
                         "same prediction. HD95 is dominated by small foreground islands "
                         "far from the pancreas, so this separates 'the network delineates "
                         "the organ better' from 'the network emits fewer distant islands'. "
                         "Both raw and filtered numbers go into one record, computed from "
                         "one inference pass, so they are paired exactly.")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.arm in METHOD:
        from run_ssl_methods_5fold import build_eval_model
        root, method = METHOD[args.arm]
        root = args.model_root or root
        fold_dir = os.path.join(root, f"fold_{args.fold}")
        model = build_eval_model(method, fold_dir)
    else:
        root, ckpt_name = SINGLE[args.arm]
        root = args.model_root or root
        ckpt = os.path.join(root, f"fold_{args.fold}", ckpt_name)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        model = get_unet(dropout_rate=0.1)
        model.load_weights(ckpt)

    pids = val_pids(root, args.fold)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_fold{args.fold}_{args.blend}" + ("_lcc" if args.also_lcc else "")
    jsonl = os.path.join(args.out_dir, f"partial_{tag}.jsonl")
    done = load_done(jsonl)
    todo = [p for p in pids if p not in done]
    print(f"{tag}: {len(pids)} volumes, {len(done)} done, {len(todo)} to do.")

    img_dir = Path(args.raw_dir) / "imagesTr"
    lbl_dir = Path(args.raw_dir) / "labelsTr"

    with open(jsonl, "a") as out:
        for pid in todo:
            ip, lp = img_dir / f"{pid}.nii.gz", lbl_dir / f"{pid}.nii.gz"
            if not ip.exists() or not lp.exists():
                print(f"  missing {pid}, skipping")
                continue

            nii = nib.load(str(ip))
            vol = nii.get_fdata().astype(np.float32)
            lbl = (nib.load(str(lp)).get_fdata() > 0).astype(np.uint8)
            # (dx, dy, dz) in mm straight from the header.
            spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])

            vol = np.clip(vol, HU_MIN, HU_MAX)
            vol = np.clip((vol - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)

            pred = np.zeros_like(vol)
            for s in range(vol.shape[2]):
                pred[:, :, s] = predict_sliding_window(model, vol[:, :, s], blend=args.blend)
            pred = (pred > 0.5).astype(np.uint8)

            def score(mask):
                d = float(compute_dice(lbl, mask))
                if mask.sum() == 0 or lbl.sum() == 0:
                    return d, None, None          # undefined, not zero
                return (d,
                        float(compute_hd95(lbl, mask, voxel_spacing=spacing)),
                        float(compute_asd(lbl, mask, voxel_spacing=spacing)))

            dice, hd, asd = score(pred)
            rec = {"case": pid, "dice": dice, "hd95": hd, "asd": asd,
                   "spacing": spacing, "arm": args.arm,
                   "fold": args.fold, "blend": args.blend}

            if args.also_lcc:
                # 26-connectivity in 3D on the reconstructed volume, never slice by
                # slice: a slice-wise largest component would sever the organ at any
                # axial gap and would not be the operation a practitioner applies.
                lab, n_comp = ndimage.label(pred, structure=np.ones((3, 3, 3)))
                if n_comp > 1:
                    sizes = ndimage.sum(pred, lab, range(1, n_comp + 1))
                    keep = int(np.argmax(sizes)) + 1
                    pred_lcc = (lab == keep).astype(np.uint8)
                else:
                    pred_lcc = pred
                d2, h2, a2 = score(pred_lcc)
                rec.update({"n_components": int(n_comp), "dice_lcc": d2,
                            "hd95_lcc": h2, "asd_lcc": a2,
                            "voxels_removed": int(pred.sum() - pred_lcc.sum())})

            out.write(json.dumps(rec) + "\n")
            out.flush()
            os.fsync(out.fileno())
            msg = f"  {pid}: Dice {dice:.4f}  HD95 {hd if hd is None else round(hd, 2)} mm"
            if args.also_lcc:
                msg += (f"  |  LCC: {rec['n_components']} comps, Dice {rec['dice_lcc']:.4f}"
                        f"  HD95 {rec['hd95_lcc'] if rec['hd95_lcc'] is None else round(rec['hd95_lcc'], 2)} mm")
            print(msg)

    rows = list(load_done(jsonl).values())
    d = [r["dice"] for r in rows]
    h = [r["hd95"] for r in rows if r["hd95"] is not None]
    a = [r["asd"] for r in rows if r["asd"] is not None]
    summary = {"arm": args.arm, "fold": args.fold, "blend": args.blend,
               "n": len(rows), "n_empty_pred": len(rows) - len(h),
               "dice_mean": float(np.mean(d)) if d else 0.0,
               "hd95_mean": float(np.mean(h)) if h else None,
               "hd95_median": float(np.median(h)) if h else None,
               "asd_mean": float(np.mean(a)) if a else None,
               "per_case": {r["case"]: r for r in rows}}
    with open(os.path.join(args.out_dir, f"hd95_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{tag}: Dice {summary['dice_mean']:.4f} | "
          f"HD95 {summary['hd95_mean']} mm | ASD {summary['asd_mean']} mm | "
          f"{summary['n_empty_pred']} empty predictions excluded from distances")


if __name__ == "__main__":
    main()
