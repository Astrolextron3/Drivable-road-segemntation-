"""
MobileNetV3-Small Attention U-Net — RGB-only (3-channel) version.

Identical architecture to model.py except:
  - in_channels defaults to 3 (no LiDAR / Radar).
  - load_from_5ch_checkpoint() performs weight surgery so the trained
    5-channel stem is collapsed to 3 channels without discarding knowledge.

Weight surgery logic
─────────────────────
Original stem conv weight shape: [16, 5, 3, 3]
  ch 0-2 → RGB weights  (keep as-is)
  ch 3   → LiDAR weight (fold into RGB channels by averaging)
  ch 4   → Radar weight (fold into RGB channels by averaging)

New stem conv weight shape: [16, 3, 3, 3]
  new_w = rgb_w + (lidar_w + radar_w) / 3   ← distributes sensor info evenly

All other layers are copied 1-to-1 (decoder, BN, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileNetSeg", "load_from_5ch_checkpoint"]


# ═══════════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS  (identical to original)
# ═══════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Hardsigmoid(inplace=True),
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.relu(self.W_x(x) + self.W_g(g_up))
        return x * self.psi(out)


class InvertedResidual(nn.Module):
    def __init__(self, in_c, exp_c, out_c, kernel, stride, use_se=False, use_hs=False):
        super().__init__()
        act = nn.Hardswish if use_hs else nn.ReLU
        self.use_skip = (stride == 1 and in_c == out_c)
        layers = []
        if exp_c != in_c:
            layers.extend([nn.Conv2d(in_c, exp_c, 1, bias=False),
                           nn.BatchNorm2d(exp_c), act(inplace=True)])
        layers.extend([nn.Conv2d(exp_c, exp_c, kernel, stride=stride,
                                 padding=kernel // 2, groups=exp_c, bias=False),
                       nn.BatchNorm2d(exp_c), act(inplace=True)])
        if use_se:
            layers.append(SEBlock(exp_c))
        layers.extend([nn.Conv2d(exp_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c)])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return (x + out) if self.use_skip else out


class MobileNetV3Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.Hardswish(inplace=True),
        )
        cfg = [
            (16,  16,  16,  3, 2, True,  False),
            (16,  72,  24,  3, 2, False, False),
            (24,  88,  24,  3, 1, False, False),
            (24,  96,  40,  5, 2, True,  True),
            (40,  240, 40,  5, 1, True,  True),
            (40,  240, 40,  5, 1, True,  True),
            (40,  120, 48,  5, 1, True,  True),
            (48,  144, 48,  5, 1, True,  True),
            (48,  288, 96,  5, 2, True,  True),
            (96,  576, 96,  5, 1, True,  True),
            (96,  576, 96,  5, 1, True,  True),
        ]
        self.layers = nn.ModuleList([InvertedResidual(*c) for c in cfg])

    def forward(self, x):
        features = []
        x = self.stem(x)
        features.append(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:  features.append(x)
            elif i == 2: features.append(x)
            elif i == 7: features.append(x)
            elif i == 10: features.append(x)
        return features


class DoubleConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)


class AttentionUNetDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.ag4 = AttentionGate(48, 96, 48)
        self.ag3 = AttentionGate(24, 48, 24)
        self.ag2 = AttentionGate(16, 24, 16)
        self.ag1 = AttentionGate(16, 16, 16)
        self.up4 = nn.ConvTranspose2d(96, 48, 2, 2);  self.conv4 = DoubleConvBlock(96, 48)
        self.up3 = nn.ConvTranspose2d(48, 24, 2, 2);  self.conv3 = DoubleConvBlock(48, 24)
        self.up2 = nn.ConvTranspose2d(24, 16, 2, 2);  self.conv2 = DoubleConvBlock(32, 16)
        self.up1 = nn.ConvTranspose2d(16, 16, 2, 2);  self.conv1 = DoubleConvBlock(32, 16)
        self.up0 = nn.ConvTranspose2d(16, 16, 2, 2)
        self.final_conv = nn.Conv2d(16, num_classes, 1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features
        x3a = self.ag4(x3, x4);  d4 = self.up4(x4)
        if d4.shape != x3a.shape: d4 = F.interpolate(d4, size=x3a.shape[2:])
        d4 = self.conv4(torch.cat([d4, x3a], 1))

        x2a = self.ag3(x2, d4);  d3 = self.up3(d4)
        if d3.shape != x2a.shape: d3 = F.interpolate(d3, size=x2a.shape[2:])
        d3 = self.conv3(torch.cat([d3, x2a], 1))

        x1a = self.ag2(x1, d3);  d2 = self.up2(d3)
        if d2.shape != x1a.shape: d2 = F.interpolate(d2, size=x1a.shape[2:])
        d2 = self.conv2(torch.cat([d2, x1a], 1))

        x0a = self.ag1(x0, d2);  d1 = self.up1(d2)
        if d1.shape != x0a.shape: d1 = F.interpolate(d1, size=x0a.shape[2:])
        d1 = self.conv1(torch.cat([d1, x0a], 1))

        return self.final_conv(self.up0(d1))


class MobileNetSeg(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = MobileNetV3Encoder(in_channels=in_channels)
        self.decoder = AttentionUNetDecoder(num_classes=num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════════════
#  WEIGHT SURGERY  —  5-ch checkpoint  →  3-ch model
# ═══════════════════════════════════════════════════════════════════════════

def load_from_5ch_checkpoint(checkpoint_path: str, device: str = "cpu") -> MobileNetSeg:
    """
    Build a 3-ch MobileNetSeg and transfer weights from a saved 5-ch checkpoint.

    Stem weight surgery:
        old shape: [16, 5, 3, 3]
        new shape: [16, 3, 3, 3]

        new_weight[:, 0:3, :, :] = old_weight[:, 0:3, :, :]          # RGB kept
                                  + (old_weight[:, 3:4, :, :] +       # LiDAR folded in
                                     old_weight[:, 4:5, :, :]) / 3    # Radar folded in
                                                                       # ÷3 = equal share

    All other layers: copied verbatim (shapes are identical).

    Args:
        checkpoint_path: Path to best_model.pth from original 5-ch training.
        device:          'cuda' or 'cpu'.

    Returns:
        3-ch MobileNetSeg with transferred + surgically-adapted weights.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    old_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model_3ch = MobileNetSeg(in_channels=3, num_classes=1).to(device)
    new_sd = model_3ch.state_dict()

    stem_key = "encoder.stem.0.weight"   # Conv2d weight in stem

    transferred, skipped = 0, 0
    for k in new_sd:
        if k not in old_sd:
            print(f"  [SKIP – not in checkpoint] {k}")
            skipped += 1
            continue

        if k == stem_key:
            # ── Weight Surgery ──
            w5 = old_sd[k]                         # [16, 5, 3, 3]
            rgb_w    = w5[:, :3, :, :]             # [16, 3, 3, 3]
            lidar_w  = w5[:, 3:4, :, :]            # [16, 1, 3, 3]
            radar_w  = w5[:, 4:5, :, :]            # [16, 1, 3, 3]
            # Distribute the extra sensor channels equally across RGB
            sensor_contrib = (lidar_w + radar_w) / 3.0   # [16, 1, 3, 3]
            w3 = rgb_w + sensor_contrib.expand_as(rgb_w)  # [16, 3, 3, 3]
            new_sd[k] = w3
            print(f"  [SURGERY] {k}: {list(w5.shape)} → {list(w3.shape)}")
        else:
            if old_sd[k].shape == new_sd[k].shape:
                new_sd[k] = old_sd[k]
                transferred += 1
            else:
                print(f"  [SHAPE MISMATCH – reinit] {k}: "
                      f"ckpt={list(old_sd[k].shape)} vs model={list(new_sd[k].shape)}")
                skipped += 1

    model_3ch.load_state_dict(new_sd)
    print(f"\n✅ Weight surgery complete.")
    print(f"   Transferred : {transferred} tensors")
    print(f"   Skipped     : {skipped} tensors")
    if "best_miou" in ckpt:
        print(f"   Source mIoU : {ckpt['best_miou']:.4f}")
    return model_3ch


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = MobileNetSeg(in_channels=3)
    dummy = torch.randn(1, 3, 256, 512)
    out = model(dummy)
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output     : {out.shape}")
