"""
Training script for MobileNetV3 + LR-ASPP drivable area segmentation.

Features:
  - Combined loss: Weighted BCE + Dice + Boundary (class imbalance)
  - Mixed-precision training (AMP)
  - OneCycleLR scheduler
  - Gradient clipping
  - Proper 2-class mIoU metric
  - Early stopping
  - Full checkpoint saving

Usage:  python train.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from model import MobileNetSeg
from dataset import DrivableDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
CONFIG = {
    "data_dir":     r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\all py codes\processed_wow",
    "save_dir":     r"C:\Users\Arindam S Katoch\Desktop\drivable_seg",
    "img_size":     (256, 512),
    "epochs":       30,
    "batch_size":   8,
    "lr":           1e-3,
    "weight_decay": 1e-3,
    "pos_weight":   2.5,        # BCE weight for drivable class
    "boundary_w":   0.3,        # boundary loss weight
    "val_split":    0.2,
    "patience":     7,          # early stopping patience
    "num_workers":  0,
    "grad_clip":    1.0,
}


# ═══════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════

# Pre-computed Sobel kernels (avoids re-creating per batch)
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3).contiguous()


def dice_loss(pred_logits, target, smooth=1.0):
    """Soft Dice loss for overlap optimization."""
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * intersection + smooth) / (union + smooth)).mean()


def boundary_loss(pred_logits, target, sobel_x, sobel_y):
    """Sobel-based boundary loss for road-edge accuracy."""
    pred_sig = torch.sigmoid(pred_logits)
    pred_edges = (torch.abs(F.conv2d(pred_sig, sobel_x, padding=1)) +
                  torch.abs(F.conv2d(pred_sig, sobel_y, padding=1)))
    gt_edges   = (torch.abs(F.conv2d(target,   sobel_x, padding=1)) +
                  torch.abs(F.conv2d(target,   sobel_y, padding=1)))
    return F.mse_loss(pred_edges, gt_edges)


class CombinedLoss(nn.Module):
    """
    L = BCE(pos_weight) + DiceLoss + boundary_w * BoundaryLoss

    - Weighted BCE: handles class imbalance (road ~ 30% of pixels)
    - Dice: directly optimizes overlap
    - Boundary: penalizes mismatched road edges
    """
    def __init__(self, pos_weight=2.5, boundary_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.boundary_weight = boundary_weight
        self.register_buffer("sobel_x", _SOBEL_X.clone())
        self.register_buffer("sobel_y", _SOBEL_Y.clone())

    def forward(self, pred, target):
        return (self.bce(pred, target)
                + dice_loss(pred, target)
                + self.boundary_weight * boundary_loss(
                    pred, target, self.sobel_x, self.sobel_y))


# ═══════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════

def compute_miou(pred_logits, target):
    """
    Mean IoU for binary segmentation (averages both classes).
    """
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    eps = 1e-6

    # Foreground (drivable) IoU
    i_fg = (pred * target).sum(dim=(2, 3))
    u_fg = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - i_fg
    iou_fg = (i_fg + eps) / (u_fg + eps)

    # Background (non-drivable) IoU
    pred_bg, target_bg = 1 - pred, 1 - target
    i_bg = (pred_bg * target_bg).sum(dim=(2, 3))
    u_bg = pred_bg.sum(dim=(2, 3)) + target_bg.sum(dim=(2, 3)) - i_bg
    iou_bg = (i_bg + eps) / (u_bg + eps)

    return ((iou_fg + iou_bg) / 2.0).mean().item()


# ═══════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════

def train():
    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ── Data ──
    full_ds = DrivableDataset(
        f"{cfg['data_dir']}/images", f"{cfg['data_dir']}/masks",
        img_size=cfg["img_size"], augment=True,
    )
    val_n = max(1, int(cfg["val_split"] * len(full_ds)))
    trn_ds, val_ds = random_split(full_ds, [len(full_ds) - val_n, val_n])

    trn_dl = DataLoader(trn_ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)

    # ── Model ──
    model = MobileNetSeg(num_classes=1).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device     : {device}")
    print(f"Dataset    : {len(full_ds)} ({len(trn_ds)} train / {val_n} val)")
    print(f"Parameters : {param_count:,}")
    print()

    # ── Loss / Optimizer / Scheduler ──
    criterion = CombinedLoss(
        pos_weight=cfg["pos_weight"],
        boundary_weight=cfg["boundary_w"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        steps_per_epoch=len(trn_dl), epochs=cfg["epochs"], pct_start=0.1,
    )
    scaler = GradScaler(enabled=use_amp)

    # ── Training loop ──
    best_miou = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_miou": [], "val_loss": []}

    for epoch in range(1, cfg["epochs"] + 1):
        # — Train —
        model.train()
        running_loss = 0.0
        pbar = tqdm(trn_dl, desc=f"Epoch {epoch:02d}/{cfg['epochs']}")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                loss = criterion(model(imgs), masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = running_loss / len(trn_dl)
        history["train_loss"].append(avg_train)

        # — Validate —
        model.eval()
        total_miou, total_vloss = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                total_miou  += compute_miou(preds, masks)
                total_vloss += criterion(preds, masks).item()

        avg_miou = total_miou / len(val_dl)
        avg_vloss = total_vloss / len(val_dl)
        history["val_miou"].append(avg_miou)
        history["val_loss"].append(avg_vloss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  >> Loss: {avg_train:.4f} | Val: {avg_vloss:.4f} | "
              f"mIoU: {avg_miou:.4f} | LR: {lr_now:.2e}")

        # — Checkpoint —
        if avg_miou > best_miou:
            best_miou = avg_miou
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_miou": best_miou,
                "config": cfg,
            }, os.path.join(cfg["save_dir"], "best_model.pth"))
            print(f"  ✅ Best mIoU = {best_miou:.4f} — saved!")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\n⏹ Early stopping (no improvement for "
                      f"{cfg['patience']} epochs)")
                break

    # ── Training curves ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_miou"], color="green")
    axes[1].set_title("Val mIoU"); axes[1].grid(True, alpha=0.3)

    axes[2].axhline(y=best_miou, color="red", linestyle="--",
                    label=f"Best: {best_miou:.4f}")
    axes[2].plot(history["val_miou"], color="green", label="mIoU")
    axes[2].set_title("Best Reference"); axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg["save_dir"], "training_curves.png"), dpi=150)
    plt.show()

    print(f"\n🏁 Done! Best mIoU: {best_miou:.4f}")
    print(f"   Model  -> {os.path.join(cfg['save_dir'], 'best_model.pth')}")


if __name__ == "__main__":
    train()