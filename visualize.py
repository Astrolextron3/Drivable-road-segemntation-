"""
Visualization utility for drivable area segmentation ground truth.

Shows side-by-side: Original | Mask | Overlay with boundary contours.

Usage:  python visualize.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
IMG_DIR   = r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\all py codes\processed_wow\images"
MASK_DIR  = r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\all py codes\processed_wow\masks"
SAVE_PATH = r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\all py codes\visualization.png"
NUM_SAMPLES   = 6
OVERLAY_COLOR = (0, 200, 0)
OVERLAY_ALPHA = 0.4


def visualize_samples(img_dir=IMG_DIR, mask_dir=MASK_DIR,
                      save_path=SAVE_PATH, n=NUM_SAMPLES):
    """Display original | mask | overlay with contours."""
    imgs  = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n]
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))[:n]

    if not imgs:
        print(f"❌ No images found in {img_dir}")
        return

    n = min(len(imgs), len(masks))
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (ip, mp) in enumerate(zip(imgs[:n], masks[:n])):
        img  = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        # Overlay
        overlay = img.copy()
        road = mask > 127
        overlay[road] = (
            overlay[road] * (1 - OVERLAY_ALPHA) +
            np.array(OVERLAY_COLOR) * OVERLAY_ALPHA
        ).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        fill_pct = road.sum() / mask.size * 100

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original [{os.path.basename(ip)}]")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(f"Mask ({fill_pct:.1f}% drivable)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay + Boundary")
        axes[i, 2].axis("off")

    plt.suptitle("Ground Truth Verification", fontsize=16, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved -> {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize_samples()