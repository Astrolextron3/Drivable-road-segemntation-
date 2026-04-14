"""
Inference script for the 3-channel RGB drivable area segmentation model.
Uses best_model_bdd.pth (fine-tuned on BDD100K, 3-ch input).

Supports:
  - Single image or full directory batch
  - FPS benchmarking
  - Green overlay + white boundary contour output
  - Side-by-side prediction grid saved as PNG

Usage:
    python predict_rgb.py
"""

import os
import time
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_rgb import MobileNetSeg

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Model ──────────────────────────────────────────────────────────────
    "model_path":  r"C:\Users\Arindam S Katoch\Desktop\drivable_seg_bdd\best_model_bdd.pth",

    # ── Input — point to any folder of .jpg/.png images ───────────────────
    # Default: use the BDD seg images (val split) for quick evaluation
    "input_dir":   r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\kys",

    # ── Output ─────────────────────────────────────────────────────────────
    "output_dir":  r"C:\Users\Arindam S Katoch\Desktop\drivable_seg_bdd\predictions",

    # ── Settings ───────────────────────────────────────────────────────────
    "img_size":    (256, 512),
    "threshold":   0.5,
    "max_images":  10,          # set None to run on entire input_dir
    "show_grid":   True,        # save side-by-side prediction grid PNG
    "benchmark":   True,        # measure FPS before inference
    "overlay_color": (0, 200, 0),
    "overlay_alpha": 0.45,
}

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, device: str) -> MobileNetSeg:
    model = MobileNetSeg(in_channels=3, num_classes=1).to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    epoch  = ckpt.get("epoch", "?")
    miou   = ckpt.get("best_miou", 0.0)
    stage  = ckpt.get("stage", "?")
    print(f"Loaded  : {os.path.basename(model_path)}")
    print(f"Epoch   : {epoch}  |  Stage: {stage}  |  Best mIoU: {miou:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  PRE / POST PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

_transform = None

def _get_transform(img_size):
    global _transform
    if _transform is None:
        _transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return _transform


def preprocess(img_rgb: np.ndarray, img_size: tuple) -> torch.Tensor:
    """Resize + normalise an RGB image. Returns [1, 3, H, W]."""
    t = _get_transform(img_size)
    return t(image=img_rgb)["image"].unsqueeze(0)


def postprocess(raw_mask: np.ndarray) -> np.ndarray:
    """Morphological cleanup + keep largest connected component."""
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  np.ones((5,  5),  np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    return mask


def create_overlay(img_rgb: np.ndarray, mask: np.ndarray,
                   color=(0, 200, 0), alpha=0.45) -> np.ndarray:
    """Green overlay + white boundary contour on the original image."""
    overlay = img_rgb.copy()
    overlay[mask == 1] = (
        overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


# ═══════════════════════════════════════════════════════════════════════════
#  BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_fps(model, device, img_size, n_runs=200):
    dummy = torch.randn(1, 3, *img_size).to(device)
    with torch.no_grad():
        for _ in range(20):           # warmup
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = n_runs / elapsed
    ms  = elapsed / n_runs * 1000
    print(f"FPS     : {fps:.1f}  ({ms:.1f} ms/frame)  "
          f"@ {img_size[1]}x{img_size[0]} on {device.upper()}")
    return fps


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict():
    cfg    = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("=" * 55)
    print("  Drivable Area Segmentation — RGB Inference")
    print("=" * 55)
    print(f"Device  : {device.upper()}")

    model = load_model(cfg["model_path"], device)

    if cfg["benchmark"]:
        print()
        benchmark_fps(model, device, cfg["img_size"])

    # ── Gather input images ────────────────────────────────────────────────
    exts  = (".jpg", ".jpeg", ".png")
    files = sorted([
        os.path.join(cfg["input_dir"], f)
        for f in os.listdir(cfg["input_dir"])
        if f.lower().endswith(exts)
    ])
    if cfg["max_images"]:
        files = files[:cfg["max_images"]]
    print(f"\nRunning inference on {len(files)} images...")
    print(f"Input   : {cfg['input_dir']}")
    print(f"Output  : {cfg['output_dir']}")
    print()

    results = []
    total_road_pct = 0.0

    with torch.no_grad():
        for fpath in files:
            fname   = os.path.basename(fpath)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                print(f"  [SKIP] Could not read {fname}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            # Preprocess → inference
            inp  = preprocess(img_rgb, cfg["img_size"]).to(device)
            pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

            # Threshold → postprocess → resize back to original resolution
            raw_mask   = (pred > cfg["threshold"]).astype(np.uint8)
            clean_mask = postprocess(raw_mask)
            mask_full  = cv2.resize(clean_mask, (w_orig, h_orig),
                                    interpolation=cv2.INTER_NEAREST)

            road_pct = mask_full.mean() * 100
            total_road_pct += road_pct

            overlay = create_overlay(img_rgb, mask_full,
                                     color=cfg["overlay_color"],
                                     alpha=cfg["overlay_alpha"])

            # Save overlay
            out_path = os.path.join(cfg["output_dir"], f"pred_{fname}")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            results.append((img_rgb, overlay, fname, road_pct))
            print(f"  {fname}  →  road {road_pct:.1f}%")

    if not results:
        print("No images processed.")
        return

    avg_road = total_road_pct / len(results)
    print(f"\nAvg road coverage : {avg_road:.1f}%")

    # ── Prediction grid ────────────────────────────────────────────────────
    if cfg["show_grid"] and results:
        n   = len(results)
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, (img, overlay, name, rpct) in enumerate(results):
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Input: {name}", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"Drivable Area  ({rpct:.1f}% road)", fontsize=9)
            axes[i, 1].axis("off")

        plt.suptitle(f"Drivable Area Segmentation  |  mIoU target: 0.85+",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        grid_path = os.path.join(cfg["output_dir"], "predictions_grid.png")
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grid saved : {grid_path}")

    print(f"\n✅ Done! Predictions saved to {cfg['output_dir']}")
    print("=" * 55)


if __name__ == "__main__":
    predict()
