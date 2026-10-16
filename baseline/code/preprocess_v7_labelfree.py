"""
Label-free patch extraction for the UNLABELED stream.
----------------------------------------------------
preprocess_v5_patches.py centres half of every patient's 70 patches on a
ground-truth pancreas pixel. It ran on all 281 patients before any split, so the
"unlabeled" stream X_u of every semi-supervised arm was organ-centred using the
very labels those patients are supposed not to have. No label-free pipeline could
produce those crops.

This script produces what a label-free pipeline can: the same loading,
orientation and HU window as v5, but crop centres drawn uniformly from BODY
voxels (raw HU > -500) on slices that contain a non-trivial amount of body. The
label file is never opened. Only *_x.npy is written, so nothing downstream can
accidentally read a label from this directory.

Seeded per patient, so the set is reproducible (v5 was unseeded).

Usage:
    python preprocess_v7_labelfree.py --raw_dir .../Task07_Pancreas \
        --output_dir .../preprocessed_v7_labelfree
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

PATCH = 256
N_PATCHES = 70            # same per-patient count as v5
HU_MIN, HU_MAX = -125, 275
BODY_HU = -500            # air/body boundary; needs no annotation
MIN_BODY_FRAC = 0.05      # skip slices that are almost all air


def extract(img_raw, rng):
    """img_raw: (D, H, W) float32 in HU."""
    D, H, W = img_raw.shape
    body = img_raw > BODY_HU
    frac = body.reshape(D, -1).mean(axis=1)
    slices = np.where(frac > MIN_BODY_FRAC)[0]
    if len(slices) == 0:
        slices = np.arange(D)

    img = np.clip(img_raw, HU_MIN, HU_MAX)
    img = np.clip((img - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)

    out = []
    for _ in range(N_PATCHES):
        s = int(rng.choice(slices))
        ys, xs = np.nonzero(body[s])
        if len(ys):
            k = int(rng.integers(len(ys)))
            cy, cx = int(ys[k]), int(xs[k])
        else:
            cy, cx = H // 2, W // 2
        y1 = min(max(0, cy - PATCH // 2), H - PATCH)
        x1 = min(max(0, cx - PATCH // 2), W - PATCH)
        out.append(img[s, y1:y1 + PATCH, x1:x1 + PATCH])
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--reference_dir",
                    default="data/preprocessed_v5_patches",
                    help="Only patients present here are extracted, so the patient set "
                         "matches v5 exactly (v5 skipped non-512x512 volumes).")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pids = sorted(p.name[:-6] for p in Path(args.reference_dir).glob("*_x.npy"))
    print(f"{len(pids)} patients to extract")

    for pid in pids:
        dst = out / f"{pid}_x.npy"
        if dst.exists():
            continue
        nii = nib.load(str(Path(args.raw_dir) / "imagesTr" / f"{pid}.nii.gz"))
        img = np.transpose(nii.get_fdata().astype(np.float32), (2, 0, 1))  # as v5
        assert img.shape[1:] == (512, 512), (pid, img.shape)
        rng = np.random.default_rng(20260928 + int(pid.split("_")[-1]))
        patches = extract(img, rng)
        tmp = out / f"{pid}_x.tmp.npy"
        np.save(tmp, patches)
        tmp.rename(dst)          # atomic: a killed job never leaves a half file
        print(f"  {pid}: {patches.shape}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
