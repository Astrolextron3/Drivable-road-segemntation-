"""
2-Stage Fine-Tuning on BDD100K for 3-channel RGB drivable area segmentation.

Loads the 5-ch nuScenes checkpoint, performs weight surgery to deflate to
3 channels, then fine-tunes in two stages:

  Stage 1 — Warm-up  (STAGE1_EPOCHS epochs, default 8)
      Encoder frozen.  Only the decoder + surgically-replaced stem are trained.
      High LR — fast adaptation without destroying deep features.

  Stage 2 — Full Fine-tune  (remaining epochs up to MAX_EPOCHS)
      Full model unfrozen.  Low LR (LR / 10).
      OneCycleLR for smooth convergence.
      Early stopping with patience.

Target: 0.85+ mIoU on BDD100K validation split.

Usage:
    python finetune_bdd.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_rgb import MobileNetSeg, load_from_5ch_checkpoint
from dataset_bdd import BDD100KDataset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # headless – safe on any machine
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit paths here
# ═══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "checkpoint_5ch": r"C:\Users\Arindam S Katoch\Desktop\drivable_seg\best_model.pth",
    # bdd_root is hardcoded inside dataset_bdd.py — no need to set here
    "save_dir":       r"C:\Users\Arindam S Katoch\Desktop\drivable_seg_bdd",

    # ── Input ──────────────────────────────────────────────────────────────
    "img_size":       (256, 512),

    # ── Training ───────────────────────────────────────────────────────────
    "stage1_epochs":  8,          # frozen encoder warm-up
    "max_epochs":     80,         # total (stage1 + stage2)
    "batch_size":     16,
    "lr":             2e-3,       # stage 1 LR (decoder only)
    "weight_decay":   1e-4,
    "boundary_w":     0.4,
    "patience":       15,         # early-stop patience (stage 2)
    "num_workers":    4,
    "grad_clip":      1.0,
}

# ═══════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS  (same as original train.py)
# ═══════════════════════════════════════════════════════════════════════════

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3).contiguous()


def focal_tversky_loss(pred_logits, target, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6):
    probs  = torch.sigmoid(pred_logits).view(-1)
    target = target.view(-1)
    tp = (probs * target).sum()
    fp = ((1 - target) * probs).sum()
    fn = (target * (1 - probs)).sum()
    tvi = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1 - tvi) ** gamma


def boundary_loss(pred_logits, target, sobel_x, sobel_y):
    p = torch.sigmoid(pred_logits)
    pe = torch.abs(F.conv2d(p, sobel_x, padding=1)) + torch.abs(F.conv2d(p, sobel_y, padding=1))
    ge = torch.abs(F.conv2d(target, sobel_x, padding=1)) + torch.abs(F.conv2d(target, sobel_y, padding=1))
    return F.mse_loss(pe, ge)


class CombinedLoss(nn.Module):
    def __init__(self, boundary_weight=0.4):
        super().__init__()
        self.bw = boundary_weight
        self.register_buffer("sx", _SOBEL_X.clone())
        self.register_buffer("sy", _SOBEL_Y.clone())

    def forward(self, pred, target):
        return focal_tversky_loss(pred, target) + self.bw * boundary_loss(pred, target, self.sx, self.sy)


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_miou(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    eps = 1e-6
    i_fg = (pred * target).sum(dim=(2, 3))
    u_fg = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - i_fg
    iou_fg = (i_fg + eps) / (u_fg + eps)
    pb, tb = 1 - pred, 1 - target
    i_bg = (pb * tb).sum(dim=(2, 3))
    u_bg = pb.sum(dim=(2, 3)) + tb.sum(dim=(2, 3)) - i_bg
    iou_bg = (i_bg + eps) / (u_bg + eps)
    return ((iou_fg + iou_bg) / 2.0).mean().item()


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def freeze_encoder(model):
    """Freeze all encoder parameters EXCEPT the new stem conv (already surgically replaced)."""
    # Freeze everything in the encoder except the stem — stem needs to adapt
    for name, param in model.encoder.named_parameters():
        if "stem" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Stage 1] Encoder frozen (stem trainable). "
          f"Trainable params: {trainable:,}")


def unfreeze_all(model):
    """Unfreeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  [Stage 2] All params unfrozen. Total: {total:,}")


def run_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_miou = 0.0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(imgs)
                loss  = criterion(preds, masks)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

            total_loss  += loss.item()
            total_miou  += compute_miou(preds, masks)

    n = len(loader)
    return total_loss / n, total_miou / n


def save_curves(history, save_dir, best_miou):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["val_miou"], color="green"); axes[1].set_title("Val mIoU"); axes[1].grid(True, alpha=0.3)
    axes[2].axhline(y=best_miou, color="red", linestyle="--", label=f"Best {best_miou:.4f}")
    axes[2].plot(history["val_miou"], color="green", label="mIoU")
    axes[2].set_title("Best Reference"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "finetune_curves.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def finetune():
    cfg = CONFIG
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ── 1. Build 3-ch model via weight surgery ──────────────────────────────
    print("=" * 60)
    print("  Weight Surgery: 5-ch → 3-ch")
    print("=" * 60)
    model = load_from_5ch_checkpoint(cfg["checkpoint_5ch"], device=device)
    print()

    # ── 2. Datasets ──────────────────────────────────────────────────────────
    print("Loading BDD100K datasets...")
    trn_ds = BDD100KDataset(split="train", img_size=cfg["img_size"], augment=True)
    val_ds = BDD100KDataset(split="val",   img_size=cfg["img_size"], augment=False)

    trn_dl = DataLoader(trn_ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=True,
                        drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)
    print(f"  Train: {len(trn_ds)} | Val: {len(val_ds)}")
    print()

    criterion = CombinedLoss(boundary_weight=cfg["boundary_w"]).to(device)
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    history   = {"train_loss": [], "val_loss": [], "val_miou": [], "stage": []}
    best_miou = 0.0
    patience_counter = 0

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 1 — Frozen encoder warm-up
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"  STAGE 1: Frozen encoder  ({cfg['stage1_epochs']} epochs)")
    print("=" * 60)
    freeze_encoder(model)

    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler_s1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_s1, max_lr=cfg["lr"],
        steps_per_epoch=len(trn_dl), epochs=cfg["stage1_epochs"],
        pct_start=0.2
    )

    for epoch in range(1, cfg["stage1_epochs"] + 1):
        pbar = tqdm(trn_dl, desc=f"[S1] Epoch {epoch:02d}/{cfg['stage1_epochs']}")

        model.train()
        running_loss = 0.0
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer_s1.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_s1)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer_s1)
            scaler.update()
            scheduler_s1.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = running_loss / len(trn_dl)

        # Validation
        model.eval()
        total_vloss, total_miou_v = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                total_vloss  += criterion(preds, masks).item()
                total_miou_v += compute_miou(preds, masks)
        avg_vloss = total_vloss / len(val_dl)
        avg_miou  = total_miou_v / len(val_dl)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_vloss)
        history["val_miou"].append(avg_miou)
        history["stage"].append(1)

        lr_now = optimizer_s1.param_groups[0]["lr"]
        print(f"  >> Loss: {avg_train:.4f} | Val: {avg_vloss:.4f} | "
              f"mIoU: {avg_miou:.4f} | LR: {lr_now:.2e}")

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "best_miou": best_miou,
                "config": cfg,
            }, os.path.join(cfg["save_dir"], "best_model_bdd.pth"))
            print(f"  ✅ Best mIoU = {best_miou:.4f} — saved!")

    # ══════════════════════════════════════════════════════════════════════
    #  STAGE 2 — Full fine-tune
    # ══════════════════════════════════════════════════════════════════════
    stage2_epochs = cfg["max_epochs"] - cfg["stage1_epochs"]
    print()
    print("=" * 60)
    print(f"  STAGE 2: Full fine-tune  ({stage2_epochs} epochs, LR = {cfg['lr']/10:.2e})")
    print("=" * 60)
    unfreeze_all(model)

    optimizer_s2 = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"] / 10.0,
        weight_decay=cfg["weight_decay"]
    )
    scheduler_s2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_s2, max_lr=cfg["lr"] / 10.0,
        steps_per_epoch=len(trn_dl), epochs=stage2_epochs,
        pct_start=0.1
    )

    for epoch in range(1, stage2_epochs + 1):
        pbar = tqdm(trn_dl, desc=f"[S2] Epoch {epoch:02d}/{stage2_epochs}")

        model.train()
        running_loss = 0.0
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer_s2.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_s2)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer_s2)
            scaler.update()
            scheduler_s2.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = running_loss / len(trn_dl)

        # Validation
        model.eval()
        total_vloss, total_miou_v = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                total_vloss  += criterion(preds, masks).item()
                total_miou_v += compute_miou(preds, masks)
        avg_vloss = total_vloss / len(val_dl)
        avg_miou  = total_miou_v / len(val_dl)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_vloss)
        history["val_miou"].append(avg_miou)
        history["stage"].append(2)

        lr_now = optimizer_s2.param_groups[0]["lr"]
        print(f"  >> Loss: {avg_train:.4f} | Val: {avg_vloss:.4f} | "
              f"mIoU: {avg_miou:.4f} | LR: {lr_now:.2e}")

        if avg_miou > best_miou:
            best_miou = avg_miou
            patience_counter = 0
            torch.save({
                "epoch": cfg["stage1_epochs"] + epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_s2.state_dict(),
                "scheduler_state_dict": scheduler_s2.state_dict(),
                "best_miou": best_miou,
                "config": cfg,
            }, os.path.join(cfg["save_dir"], "best_model_bdd.pth"))
            print(f"  ✅ Best mIoU = {best_miou:.4f} — saved!")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\n⏹ Early stopping ({cfg['patience']} epochs without improvement).")
                break

        if best_miou >= 0.85:
            print(f"\n🎯 Target mIoU 0.85 reached ({best_miou:.4f})!")
            # keep going anyway to squeeze more out unless near end
            pass

    # ── Final curves ──────────────────────────────────────────────────────
    save_curves(history, cfg["save_dir"], best_miou)

    # Mark stage boundary on curve
    s1_end = cfg["stage1_epochs"]
    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete!")
    print(f"  Best mIoU : {best_miou:.4f}")
    print(f"  Model     : {os.path.join(cfg['save_dir'], 'best_model_bdd.pth')}")
    print(f"  Curves    : {os.path.join(cfg['save_dir'], 'finetune_curves.png')}")
    print(f"  Stage 1   : epochs 1–{s1_end}  (frozen encoder)")
    print(f"  Stage 2   : epochs {s1_end+1}+  (full fine-tune)")
    print(f"{'='*60}")


if __name__ == "__main__":
    finetune()
