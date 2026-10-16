"""
Compute 95th Percentile Hausdorff Distance (HD95)
--------------------------------------------------
For each test volume, computes HD95 between predicted and ground truth
segmentation surfaces. Also computes Dice for verification.

Usage:
    python baseline/code/compute_hd95.py \
        --image_dir /path/to/images \
        --label_dir /path/to/labels \
        --model_path baseline/models/model_patch_best.h5 \
        --output_dir baseline/logs/hd95/ \
        --model_type binary
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from datetime import datetime
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from tqdm import tqdm

# Defer TF import until needed (for CPU-only HD95 on saved predictions)
tf = None
layers = None


def lazy_import_tf():
    """Import TensorFlow only when needed for inference."""
    global tf, layers
    if tf is None:
        import tensorflow as _tf
        from tensorflow.keras import layers as _layers
        tf = _tf
        layers = _layers


def compute_hd95(y_true, y_pred, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute 95th percentile Hausdorff Distance between two binary volumes.

    Parameters
    ----------
    y_true : np.ndarray, binary ground truth mask
    y_pred : np.ndarray, binary predicted mask
    voxel_spacing : tuple, voxel spacing in mm (z, y, x) or (x, y, z)

    Returns
    -------
    hd95 : float, 95th percentile Hausdorff distance in mm
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    # Handle empty masks
    if not np.any(y_true) and not np.any(y_pred):
        return 0.0
    if not np.any(y_true) or not np.any(y_pred):
        return float('inf')

    # Extract surface voxels via erosion
    struct = generate_binary_structure(3, 1)
    y_true_border = y_true ^ binary_erosion(y_true, struct)
    y_pred_border = y_pred ^ binary_erosion(y_pred, struct)

    # Get surface point coordinates in physical space
    true_pts = np.argwhere(y_true_border) * np.array(voxel_spacing)
    pred_pts = np.argwhere(y_pred_border) * np.array(voxel_spacing)

    if len(true_pts) == 0 or len(pred_pts) == 0:
        return float('inf')

    # Compute symmetric surface distances using KD-trees
    tree_true = cKDTree(true_pts)
    tree_pred = cKDTree(pred_pts)

    dist_pred_to_true, _ = tree_true.query(pred_pts)
    dist_true_to_pred, _ = tree_pred.query(true_pts)

    all_distances = np.concatenate([dist_pred_to_true, dist_true_to_pred])
    return np.percentile(all_distances, 95)


def compute_dice(y_true, y_pred):
    """Compute Dice coefficient between two binary volumes."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    if np.sum(y_true) + np.sum(y_pred) == 0:
        return 1.0
    return 2.0 * intersection / (np.sum(y_true) + np.sum(y_pred))


def compute_asd(y_true, y_pred, voxel_spacing=(1.0, 1.0, 1.0)):
    """Compute Average Surface Distance (ASD) between two binary volumes."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    if not np.any(y_true) and not np.any(y_pred):
        return 0.0
    if not np.any(y_true) or not np.any(y_pred):
        return float('inf')

    struct = generate_binary_structure(3, 1)
    y_true_border = y_true ^ binary_erosion(y_true, struct)
    y_pred_border = y_pred ^ binary_erosion(y_pred, struct)

    true_pts = np.argwhere(y_true_border) * np.array(voxel_spacing)
    pred_pts = np.argwhere(y_pred_border) * np.array(voxel_spacing)

    if len(true_pts) == 0 or len(pred_pts) == 0:
        return float('inf')

    tree_true = cKDTree(true_pts)
    tree_pred = cKDTree(pred_pts)

    dist_pred_to_true, _ = tree_true.query(pred_pts)
    dist_true_to_pred, _ = tree_pred.query(true_pts)

    return (np.mean(dist_pred_to_true) + np.mean(dist_true_to_pred)) / 2.0


# --- U-Net model definition (must match training) ---
def get_unet(img_size=256, num_classes=1):
    lazy_import_tf()
    inputs = layers.Input(shape=(img_size, img_size, 1))
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    if num_classes == 1:
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    model = tf.keras.Model(inputs, outputs)
    return model


def sliding_window_predict(model, volume, patch_size=256, stride=128, num_classes=1):
    """Run sliding window inference on a single 3D volume (slice-by-slice)."""
    D, H, W = volume.shape
    pred_volume = np.zeros((D, H, W), dtype=np.float32)
    count_volume = np.zeros((D, H, W), dtype=np.float32)

    for d in range(D):
        slc = volume[d]
        # Pad if needed
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        if pad_h > 0 or pad_w > 0:
            slc = np.pad(slc, ((0, pad_h), (0, pad_w)), mode='constant')

        H_pad, W_pad = slc.shape

        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = slc[y:y+patch_size, x:x+patch_size]
                patch_input = patch[np.newaxis, :, :, np.newaxis]
                pred_patch = model.predict(patch_input, verbose=0)

                if num_classes == 1:
                    pred_patch = pred_patch[0, :, :, 0]
                else:
                    # For multiclass, take argmax
                    pred_patch = np.argmax(pred_patch[0], axis=-1).astype(np.float32)

                # Only write back within original bounds
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                pred_volume[d, y:y_end, x:x_end] += pred_patch[:y_end-y, :x_end-x]
                count_volume[d, y:y_end, x:x_end] += 1.0

    # Average overlapping predictions (for binary; for multiclass this is approximate)
    mask = count_volume > 0
    pred_volume[mask] /= count_volume[mask]
    return pred_volume


def process_volume_from_nifti(image_path, label_path, model=None, pred_path=None,
                               model_type='binary', patch_size=256, stride=128):
    """
    Process a single volume: load image/label, run inference or load predictions,
    compute HD95 and Dice.

    Parameters
    ----------
    image_path : str, path to image NIfTI
    label_path : str, path to ground truth label NIfTI
    model : keras model or None
    pred_path : str or None, path to pre-saved prediction NIfTI
    model_type : 'binary' or 'multiclass'
    """
    # Load ground truth
    label_nii = nib.load(label_path)
    label = label_nii.get_fdata()
    voxel_spacing = label_nii.header.get_zooms()[:3]

    # For binary: binarize label (any nonzero -> 1)
    if model_type == 'binary':
        label_binary = (label > 0).astype(np.uint8)
    else:
        label_binary = label.astype(np.uint8)

    # Get prediction
    if pred_path and os.path.exists(pred_path):
        pred_nii = nib.load(pred_path)
        pred = pred_nii.get_fdata()
    elif model is not None:
        # Load image and run inference
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()
        # HU windowing and normalization
        image = np.clip(image, -125, 275)
        image = (image - (-125)) / (275 - (-125))

        # Ensure (D, H, W) ordering
        if image.ndim == 3:
            # NIfTI is typically (H, W, D), transpose to (D, H, W)
            image = np.transpose(image, (2, 0, 1))
            label_binary = np.transpose(label_binary, (2, 0, 1))

        num_classes = 1 if model_type == 'binary' else 3
        pred = sliding_window_predict(model, image, patch_size, stride, num_classes)

        if model_type == 'binary':
            pred = (pred > 0.5).astype(np.uint8)

        # Transpose back to NIfTI ordering
        pred = np.transpose(pred, (1, 2, 0))
        label_binary = np.transpose(label_binary, (1, 2, 0))
    else:
        raise ValueError("Must provide either model or pred_path")

    if model_type == 'binary':
        pred_binary = (pred > 0.5).astype(np.uint8) if pred.max() <= 1.0 else (pred > 0).astype(np.uint8)
    else:
        pred_binary = pred.astype(np.uint8)

    # Compute metrics
    if model_type == 'binary':
        dice = compute_dice(label_binary, pred_binary)
        hd95 = compute_hd95(label_binary, pred_binary, voxel_spacing)
        asd = compute_asd(label_binary, pred_binary, voxel_spacing)
        return {'dice': dice, 'hd95': hd95, 'asd': asd}
    else:
        # Per-class metrics (skip background class 0)
        results = {}
        unique_labels = np.unique(label_binary)
        for cls in unique_labels:
            if cls == 0:
                continue
            gt_cls = (label_binary == cls).astype(np.uint8)
            pred_cls = (pred_binary == cls).astype(np.uint8)
            results[f'class_{int(cls)}_dice'] = compute_dice(gt_cls, pred_cls)
            results[f'class_{int(cls)}_hd95'] = compute_hd95(gt_cls, pred_cls, voxel_spacing)
            results[f'class_{int(cls)}_asd'] = compute_asd(gt_cls, pred_cls, voxel_spacing)
        return results


def main():
    parser = argparse.ArgumentParser(description='Compute HD95 for segmentation predictions')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing image NIfTI files')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory containing label NIfTI files')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model .h5 file (for inference)')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Directory containing pre-saved prediction NIfTI files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--model_type', type=str, default='binary',
                        choices=['binary', 'multiclass'],
                        help='Model type: binary or multiclass')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--model_name', type=str, default='model',
                        help='Name tag for output files')
    parser.add_argument('--cases', type=str, nargs='+', default=None,
                        help='Specific case names to process (e.g., pancreas_001 pancreas_004). If None, process all.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load model if needed
    model = None
    if args.model_path and not args.pred_dir:
        lazy_import_tf()
        print(f"Loading model from {args.model_path}")
        num_classes = 1 if args.model_type == 'binary' else 3
        model = get_unet(img_size=args.patch_size, num_classes=num_classes)
        model.load_weights(args.model_path)
        print("Model loaded successfully.")

    # Find label files
    label_files = sorted(glob.glob(os.path.join(args.label_dir, '*.nii.gz')))
    if not label_files:
        label_files = sorted(glob.glob(os.path.join(args.label_dir, '*.nii')))
    print(f"Found {len(label_files)} label files")

    all_results = []

    # Filter to specific cases if requested
    if args.cases:
        label_files = [f for f in label_files
                       if any(c in os.path.basename(f) for c in args.cases)]
        print(f"Filtered to {len(label_files)} cases: {args.cases}")

    for label_path in tqdm(label_files, desc="Computing HD95"):
        case_name = os.path.basename(label_path).replace('.nii.gz', '').replace('.nii', '')
        # Remove 'label_' or similar prefixes to match image name
        image_name = case_name.replace('label_', '').replace('_label', '')

        # Find corresponding image
        image_path = None
        for ext in ['.nii.gz', '.nii']:
            candidate = os.path.join(args.image_dir, image_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
            # Try with common prefixes
            for prefix in ['', 'image_', 'img_']:
                candidate = os.path.join(args.image_dir, prefix + image_name + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

        # Find pre-saved prediction if available
        pred_path = None
        if args.pred_dir:
            for ext in ['.nii.gz', '.nii']:
                candidate = os.path.join(args.pred_dir, case_name + ext)
                if os.path.exists(candidate):
                    pred_path = candidate
                    break
                candidate = os.path.join(args.pred_dir, image_name + ext)
                if os.path.exists(candidate):
                    pred_path = candidate
                    break

        if image_path is None and model is not None:
            print(f"  WARNING: No image found for {case_name}, skipping.")
            continue

        if model is None and pred_path is None:
            print(f"  WARNING: No model or prediction for {case_name}, skipping.")
            continue

        try:
            result = process_volume_from_nifti(
                image_path=image_path,
                label_path=label_path,
                model=model,
                pred_path=pred_path,
                model_type=args.model_type,
                patch_size=args.patch_size,
                stride=args.stride,
            )
            result['case'] = case_name
            all_results.append(result)
            if args.model_type == 'binary':
                print(f"  {case_name}: Dice={result['dice']:.4f}, HD95={result['hd95']:.2f}mm, ASD={result['asd']:.2f}mm")
            else:
                for key, val in sorted(result.items()):
                    if key != 'case':
                        print(f"  {case_name} {key}: {val:.4f}")
        except Exception as e:
            print(f"  ERROR processing {case_name}: {e}")
            all_results.append({'case': case_name, 'dice': float('nan'), 'hd95': float('nan'), 'asd': float('nan')})

    # --- Write report ---
    output_file = os.path.join(args.output_dir, f'hd95_results_{args.model_name}_{timestamp}.txt')

    with open(output_file, 'w') as f:
        f.write(f"HD95 Results: {args.model_name}\n")
        f.write(f"Model: {args.model_path or 'pre-saved predictions'}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        if args.model_type == 'binary':
            dices = [r['dice'] for r in all_results if not np.isnan(r.get('dice', float('nan')))]
            hd95s = [r['hd95'] for r in all_results if not np.isnan(r.get('hd95', float('nan'))) and r.get('hd95', float('inf')) < float('inf')]
            asds = [r['asd'] for r in all_results if not np.isnan(r.get('asd', float('nan'))) and r.get('asd', float('inf')) < float('inf')]

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"N volumes:       {len(all_results)}\n")
            if dices:
                f.write(f"Dice mean (std): {np.mean(dices):.4f} ({np.std(dices):.4f})\n")
            if hd95s:
                f.write(f"HD95 mean (std): {np.mean(hd95s):.2f} ({np.std(hd95s):.2f}) mm\n")
                f.write(f"HD95 median:     {np.median(hd95s):.2f} mm\n")
            if asds:
                f.write(f"ASD  mean (std): {np.mean(asds):.2f} ({np.std(asds):.2f}) mm\n")
            f.write("\n")

            f.write("PER-CASE RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Case':<30} {'Dice':>8} {'HD95 (mm)':>12} {'ASD (mm)':>12}\n")
            f.write("-" * 62 + "\n")
            for r in all_results:
                dice_str = f"{r['dice']:.4f}" if not np.isnan(r.get('dice', float('nan'))) else "N/A"
                hd95_str = f"{r['hd95']:.2f}" if r.get('hd95', float('inf')) < float('inf') else "inf"
                asd_str = f"{r['asd']:.2f}" if r.get('asd', float('inf')) < float('inf') else "inf"
                f.write(f"{r['case']:<30} {dice_str:>8} {hd95_str:>12} {asd_str:>12}\n")
        else:
            f.write("PER-CASE MULTICLASS RESULTS\n")
            f.write("-" * 40 + "\n")
            for r in all_results:
                f.write(f"\n{r['case']}:\n")
                for key, val in sorted(r.items()):
                    if key != 'case':
                        f.write(f"  {key}: {val:.4f}\n")

    print(f"\nResults saved to: {output_file}")

    # Print summary
    if args.model_type == 'binary' and all_results:
        dices = [r['dice'] for r in all_results if not np.isnan(r.get('dice', float('nan')))]
        hd95s = [r['hd95'] for r in all_results if r.get('hd95', float('inf')) < float('inf')]
        print(f"\n{'='*50}")
        print(f"SUMMARY: {args.model_name}")
        print(f"{'='*50}")
        if dices:
            print(f"Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
        if hd95s:
            print(f"HD95: {np.mean(hd95s):.2f} +/- {np.std(hd95s):.2f} mm")


if __name__ == '__main__':
    main()
