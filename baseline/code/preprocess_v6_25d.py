"""
Preprocessing V6: 2.5D patch extraction (neighbour-slice context)
-----------------------------------------------------------------
Re-extracts the same kind of 256x256 balanced patches as
`preprocess_v5_patches.py`, but stores a stack of NEIGHBOURING slices per
patch instead of a single slice:

    x: (N, 256, 256, 5)  float32   slices [s-2, s-1, s, s+1, s+2]
    y: (N, 256, 256)     uint8     label of the CENTRE slice s only

Edge slices are replicated when s is near the volume boundary, so the centre
slice is always at channel index 2.

This one extraction serves every arm of the 2.5D ablation:
    n_slices=1  -> channel [2]        (matched 2D baseline)
    n_slices=3  -> channels [1:4]
    n_slices=5  -> channels [0:5]

Unlike v5 (which used an unseeded RNG), patch coordinates here are drawn from
a per-case seeded RNG and written to `patch_meta.json`, so every arm of the
ablation trains on the identical patch set and the comparison is controlled.

Usage (single shard of a SLURM array):
    python preprocess_v6_25d.py --raw_dir <Task07> --output_dir <out> \
        --shard 0 --n_shards 10
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

PATCH_SIZE = 256
N_PATCHES_PER_VOL = 70      # matches the existing v5 arrays (70 per patient)
CONTEXT = 2                 # +/- 2 slices -> 5 channels
HU_MIN, HU_MAX = -125, 275
BASE_SEED = 42


def stack_slices(img_vol, slice_idx, y1, x1, context=CONTEXT, patch_size=PATCH_SIZE):
    """Crop the same (y1, x1) window from slices slice_idx-context .. +context.

    img_vol is (D, H, W). Out-of-range slice indices are clamped to the volume
    boundary (edge replication), so the centre slice stays at channel `context`.
    """
    depth = img_vol.shape[0]
    chans = []
    for offset in range(-context, context + 1):
        s = min(max(slice_idx + offset, 0), depth - 1)
        chans.append(img_vol[s, y1:y1 + patch_size, x1:x1 + patch_size])
    return np.stack(chans, axis=-1)


def extract_patches(img_vol, label_vol, rng,
                    patch_size=PATCH_SIZE, n_patches_per_vol=N_PATCHES_PER_VOL):
    """Balanced 50/50 foreground/background sampling, mirroring v5's strategy."""
    patches_x, patches_y, meta = [], [], []

    depth, H, W = img_vol.shape
    fg_indices = [i for i in range(depth) if label_vol[i].sum() > 0]
    if len(fg_indices) == 0:
        print("  Warning: no pancreas in volume, falling back to all slices.")
        fg_indices = list(range(depth))

    num_fg = n_patches_per_vol // 2

    # --- foreground-centred patches ---
    tries = 0
    while len(patches_x) < num_fg and tries < num_fg * 5:
        tries += 1
        slice_idx = int(rng.choice(fg_indices))

        ys, xs = np.where(label_vol[slice_idx] > 0)
        if len(ys) > 0:
            center_y = int(rng.choice(ys))
            center_x = int(rng.choice(xs))
        else:
            center_y, center_x = H // 2, W // 2

        jitter = patch_size // 4
        center_y += int(rng.integers(-jitter, jitter))
        center_x += int(rng.integers(-jitter, jitter))

        y1 = max(0, center_y - patch_size // 2)
        x1 = max(0, center_x - patch_size // 2)
        if y1 + patch_size > H:
            y1 = H - patch_size
        if x1 + patch_size > W:
            x1 = W - patch_size
        y1, x1 = max(0, y1), max(0, x1)

        img_patch = stack_slices(img_vol, slice_idx, y1, x1)
        if img_patch.shape[:2] == (patch_size, patch_size):
            patches_x.append(img_patch)
            patches_y.append(label_vol[slice_idx, y1:y1 + patch_size, x1:x1 + patch_size])
            meta.append({"slice": slice_idx, "y1": int(y1), "x1": int(x1), "kind": "fg"})

    # --- background patches ---
    tries = 0
    while len(patches_x) < n_patches_per_vol and tries < n_patches_per_vol * 5:
        tries += 1
        slice_idx = int(rng.integers(0, depth))
        y1 = int(rng.integers(0, H - patch_size))
        x1 = int(rng.integers(0, W - patch_size))

        img_patch = stack_slices(img_vol, slice_idx, y1, x1)
        if img_patch.shape[:2] == (patch_size, patch_size):
            patches_x.append(img_patch)
            patches_y.append(label_vol[slice_idx, y1:y1 + patch_size, x1:x1 + patch_size])
            meta.append({"slice": slice_idx, "y1": int(y1), "x1": int(x1), "kind": "bg"})

    return np.array(patches_x, dtype=np.float32), np.array(patches_y, dtype=np.uint8), meta


def process_case(case_id, case_index, img_dir, label_dir, out_dir, n_patches):
    out_x = out_dir / f"{case_id}_x.npy"
    if out_x.exists():
        print(f"  {case_id}: already done, skipping.")
        return

    img_path = img_dir / f"{case_id}.nii.gz"
    lab_path = label_dir / f"{case_id}.nii.gz"
    if not img_path.exists() or not lab_path.exists():
        print(f"  {case_id}: NIfTI missing, skipping.")
        return

    img = nib.load(str(img_path)).get_fdata().astype(np.float32)
    lab = nib.load(str(lab_path)).get_fdata().astype(np.uint8)

    # (H, W, D) -> (D, H, W), same convention as v5
    img = np.transpose(img, (2, 0, 1))
    lab = np.transpose(lab, (2, 0, 1))

    if img.shape[1] != 512 or img.shape[2] != 512:
        print(f"  {case_id}: non-standard size {img.shape}, skipping.")
        return

    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    img = np.clip(img, 0.0, 1.0)

    # Deterministic per-case seed: same patches on every re-run and every arm.
    rng = np.random.default_rng(BASE_SEED + case_index)
    px, py, meta = extract_patches(img, lab, rng, n_patches_per_vol=n_patches)

    np.save(out_dir / f"{case_id}_y.npy", py)
    np.save(out_x, px)  # x written last: its presence marks the case as complete
    with open(out_dir / f"{case_id}_meta.json", "w") as f:
        json.dump({"case": case_id, "seed": BASE_SEED + case_index,
                   "n_patches": int(len(px)), "context": CONTEXT,
                   "depth": int(img.shape[0]), "patches": meta}, f)
    print(f"  {case_id}: {px.shape} patches saved.")


def main():
    parser = argparse.ArgumentParser(description="2.5D patch extraction (v6).")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Task07_Pancreas root (contains imagesTr/, labelsTr/)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--n_patches", type=int, default=N_PATCHES_PER_VOL)
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    img_dir, label_dir = raw_path / "imagesTr", raw_path / "labelsTr"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Case list must be the SAME 281 patients as preprocessed_v5_patches, in the
    # same sorted order, so KFold(seed=42) reproduces the existing fold splits.
    cases = sorted(p.name.replace(".nii.gz", "") for p in img_dir.glob("*.nii.gz")
                   if not p.name.startswith("."))
    print(f"Found {len(cases)} cases; shard {args.shard}/{args.n_shards}")

    for idx, case_id in enumerate(tqdm(cases)):
        if idx % args.n_shards != args.shard:
            continue
        try:
            process_case(case_id, idx, img_dir, label_dir, out_dir, args.n_patches)
        except Exception as exc:
            print(f"  Error on {case_id}: {exc}")

    print("Shard complete.")


if __name__ == "__main__":
    main()
