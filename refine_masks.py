"""
Post-hoc Mask Refinement for already-generated masks.

Run this AFTER data_prep.py if you want an additional refinement pass,
or on masks from any other source that need cleanup.

Pipeline:
  1. Morphological Close  — fill holes (lane lines, shadows, cracks)
  2. Morphological Open   — remove small floating noise blobs
  3. Horizon Guard        — erase top portion (sky/buildings)
  4. Texture Guard        — remove high-texture areas (grass, gravel)
  5. Edge Snapping        — cut mask at physical curbs using Canny edges
  6. Smoothing            — Gaussian blur + re-threshold for smooth boundaries
  7. Largest Component    — keep only the main road region

Usage (from Anaconda Prompt):
    python refine_masks.py
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
MASK_DIR     = r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\all py codes\processed_wow\masks"
IMG_DIR      = r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\all py codes\processed_wow\images"
HORIZON_FRAC = 0.40     # mask out top 40%
FILL_KERNEL  = 15       # kernel size for hole filling
CLEAN_KERNEL = 5        # kernel size for noise removal
EDGE_DILATE  = 2        # dilation iterations for edge zone
TEXTURE_THRESH = 25     # Laplacian variance threshold for "textured" regions
SMOOTH_KERNEL  = 5      # Gaussian blur kernel for boundary smoothing


def refine_mask(mask, img_bgr=None):
    """
    Apply multi-stage refinement to a single mask.

    Args:
        mask:    Single-channel uint8 mask (0/255).
        img_bgr: Corresponding BGR image (enables edge & texture refinement).

    Returns:
        Refined mask (uint8, 0/255).
    """
    h, w = mask.shape

    # ── 1. Fill holes (lane markings, shadows, small gaps) ──
    k_fill = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FILL_KERNEL, FILL_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_fill)

    # ── 2. Remove small noise blobs ──
    k_clean = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLEAN_KERNEL, CLEAN_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_clean)

    # ── 3. Horizon guard — road doesn't exist in the sky ──
    horizon_line = int(h * HORIZON_FRAC)
    mask[:horizon_line, :] = 0

    if img_bgr is not None:
        # ── 4. Texture guard — remove high-texture regions ──
        # Roads are smooth; grass/gravel/buildings are textured.
        # Use local Laplacian variance to detect texture.
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(gray_smooth, cv2.CV_64F)

        # Compute local variance in 15x15 windows
        lap_abs = np.abs(laplacian).astype(np.float32)
        local_var = cv2.blur(lap_abs, (15, 15))

        # High-texture pixels are likely NOT road
        texture_mask = (local_var > TEXTURE_THRESH).astype(np.uint8) * 255
        mask[texture_mask > 0] = 0

        # ── 5. Edge snapping — cut at physical curbs/barriers ──
        gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray_filtered, 50, 150)
        edge_zone = cv2.dilate(
            edges, np.ones((3, 3), np.uint8), iterations=EDGE_DILATE
        )
        mask[edge_zone > 0] = 0

    # ── 6. Smooth boundaries ──
    # Gaussian blur + re-threshold gives clean, smooth mask edges
    mask_smooth = cv2.GaussianBlur(mask, (SMOOTH_KERNEL, SMOOTH_KERNEL), 0)
    mask = (mask_smooth > 127).astype(np.uint8) * 255

    # Re-close after edge cutting to repair small breaks
    k_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_repair)

    # ── 7. Keep only the largest connected component ──
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    return mask


def refine_all():
    """Refine all masks in-place."""
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))
    if not mask_files:
        print(f"❌ No masks found in {MASK_DIR}")
        return

    print(f"Refining {len(mask_files)} masks...")
    print(f"  Horizon guard : top {HORIZON_FRAC*100:.0f}%")
    print(f"  Texture guard : Laplacian variance > {TEXTURE_THRESH}")
    print(f"  Edge snapping : Canny + {EDGE_DILATE}px dilation")
    print()

    stats = {"improved": 0, "unchanged": 0}

    for mp in tqdm(mask_files, desc="Refining"):
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        original_sum = mask.sum()

        # Load corresponding image for edge/texture refinement
        name = os.path.basename(mp).replace(".png", ".jpg")
        img_path = os.path.join(IMG_DIR, name)
        img_bgr = cv2.imread(img_path) if os.path.exists(img_path) else None

        refined = refine_mask(mask, img_bgr)
        cv2.imwrite(mp, refined)

        if abs(refined.sum() - original_sum) > 1000:
            stats["improved"] += 1
        else:
            stats["unchanged"] += 1

    print(f"\n✅ Refinement complete!")
    print(f"   Modified  : {stats['improved']} masks")
    print(f"   Unchanged : {stats['unchanged']} masks")


if __name__ == "__main__":
    refine_all()