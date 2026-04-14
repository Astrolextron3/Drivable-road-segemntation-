# **Drivable Area Segmentation: Real-Time MobileNetV3 + LR-ASPP**

## **Overview**
Drivable area segmentation is a crucial task for autonomous driving systems, enabling vehicles to distinguish between drivable and non-drivable regions. This repository implements a high-performance real-time segmentation model using **MobileNetV3-Small** and **Lite R-ASPP** decoder. It is trained on the **nuScenes** dataset and fine-tuned on **BDD100K** for better generalization, achieving an impressive **0.8516 mIoU**. The model is optimized for real-time inference (~30 FPS) and supports both **RGB-only** and **RGB+LiDAR+Radar** input configurations.

---

## **Project Summary**

**Author**: Arindam Sushil Katoch  
**Challenge**: MAHE-MOBILITY CHALLENGE/2026 (Team Submission)  
**Role**: Full-stack development (model architecture, training pipeline, data preprocessing, evaluation)

### **Key Features**
- **mIoU**: 0.8516 on BDD100K validation set
- **Model Size**: ~1.1 million parameters, significantly smaller than ResNet50 (125×)
- **Inference Speed**: 70+ FPS on GPU, optimized for embedded systems
- **Training Pipeline**: Multi-stage pipeline (nuScenes → BDD100K)
- **Advanced Loss Functions**: Weighted BCE, Dice, and Sobel boundary loss
- **Multi-sensor Fusion**: Supports both RGB-only and multi-channel (RGB+LiDAR+Radar) inputs with intelligent weight surgery
- **Robust Data Pipeline**: 7-stage mask generation and morphological refinement for accurate segmentation

---

## **Technical Stack**
- **Framework**: PyTorch 2.0+
- **Backbone**: MobileNetV3-Small (lightweight, efficient CNN)
- **Decoder**: Lite R-ASPP + Attention U-Net skip connections
- **Training Optimization**: Mixed-precision training (AMP), gradient clipping, OneCycleLR scheduler
- **Datasets**: nuScenes v1.0-mini, BDD100K
- **Data Augmentation**: Albumentations (geometric and photometric)

---

## **Repository Structure**

```plaintext
# Data Processing
data_prep.py          → Image-mask generation from nuScenes HD maps
refine_masks.py       → Morphological refinement for mask post-processing
visualize.py          → Ground truth visualization for comparison

# Model Architecture
model.py              → 5-channel (RGB+LiDAR+Radar) segmentation model
model_rgb.py          → 3-channel RGB-only model with weight surgery
dataset.py            → nuScenes dataset loader
dataset_bdd.py        → BDD100K dataset loader

# Training & Inference
train.py              → Base model training on nuScenes dataset (~30 epochs)
finetune_bdd.py       → Fine-tuning on BDD100K (~80 epochs)
predict.py            → Inference and batch processing with visualizations

# Pre-trained Checkpoint
best_model_bdd.pth    → Fine-tuned model achieving 0.8516 mIoU on BDD100K
