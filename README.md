Drivable Area Segmentation: Real-Time MobileNetV3 + LR-ASPP
Overview

Drivable area segmentation is a crucial task for autonomous driving systems, enabling vehicles to distinguish between drivable and non-drivable regions. This repository implements a high-performance real-time segmentation model using MobileNetV3-Small and Lite R-ASPP decoder. It is trained on the nuScenes dataset and fine-tuned on BDD100K for better generalization, achieving an impressive 0.8516 mIoU. The model is optimized for real-time inference (~30 FPS) and supports both RGB-only and RGB+LiDAR+Radar input configurations.

Project Summary

Author: Arindam Sushil Katoch
Challenge: MAHE-MOBILITY CHALLENGE/2026 (Team Submission)
Role: Full-stack development (model architecture, training pipeline, data preprocessing, evaluation)

Key Features
mIoU: 0.8516 on BDD100K validation set
Model Size: ~1.1 million parameters, significantly smaller than ResNet50 (125×)
Inference Speed: 70+ FPS on GPU, optimized for embedded systems
Training Pipeline: Multi-stage pipeline (nuScenes → BDD100K)
Advanced Loss Functions: Weighted BCE, Dice, and Sobel boundary loss
Multi-sensor Fusion: Supports both RGB-only and multi-channel (RGB+LiDAR+Radar) inputs with intelligent weight surgery
Robust Data Pipeline: 7-stage mask generation and morphological refinement for accurate segmentation
Technical Stack
Framework: PyTorch 2.0+
Backbone: MobileNetV3-Small (lightweight, efficient CNN)
Decoder: Lite R-ASPP + Attention U-Net skip connections
Training Optimization: Mixed-precision training (AMP), gradient clipping, OneCycleLR scheduler
Datasets: nuScenes v1.0-mini, BDD100K
Data Augmentation: Albumentations (geometric and photometric)
Repository Structure
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
Quick Start Guide
Installation

Install the required dependencies:

pip install torch torchvision albumentations opencv-python tqdm nuscenes
Data Preparation

Prepare the nuScenes data for training:

python data_prep.py
Training

Train the base model on nuScenes:

python train.py
Fine-Tuning

Fine-tune the model on BDD100K:

python finetune_bdd.py
Inference

Run inference and visualize results:

python predict.py
Performance Metrics
Metric	Value
mIoU (BDD100K)	0.8516
Drivable Class IoU	~87.5%
Non-drivable Class IoU	~82.8%
Model Parameters	1.1M
FPS (GPU)	70+ FPS
Inference Time	33-40 ms
Technical Innovations
Geometric Projection: Efficient transformation from 3D HD map polygons to 2D camera images for better segmentation accuracy.
GrabCut Refinement: Interactive segmentation based on projected masks for precise boundary detection.
Boundary Loss: Sobel-based penalty function incorporated to enhance accuracy near road boundaries.
Weight Surgery: Conversion of a 5-channel model to a 3-channel model without any performance loss.
Two-Stage Fine-tuning: First, a warm-up with frozen encoder layers followed by full model fine-tuning to improve convergence.
Model Architecture Overview

The model consists of an encoder-decoder architecture for efficient semantic segmentation:

Input Image [B, 3, H, W]
      ↓
MobileNetV3-Small Encoder
  (11 × Inverted Residual Blocks)
      ↓
Lite R-ASPP Module
  (Multi-scale parallel pooling)
      ↓
Attention U-Net Decoder
  (Transposed convolution + attention gates)
      ↓
Output Logit Map [B, 1, H, W]
      ↓
Sigmoid + Threshold (0.5)
      ↓
Binary Mask [B, 1, H, W]
Use Cases
Autonomous Vehicle Navigation: Identifying drivable paths for safe vehicle navigation.
Drivable Surface Detection: Detecting roads, parking lots, and driveways.
Scene Understanding: Assisting autonomous vehicles in perceiving and navigating the environment.
Mobile Robotics: Enabling mobile robots to avoid obstacles and find clear paths.
Drone Landing Zones: Identifying safe landing zones for autonomous drones.
Advanced Features
Data Pipeline:
3D HD map integration with nuScenes for geometric transformations (world → ego → camera)
Multi-stage mask refinement (7 stages) using morphological operations for better segmentation.
Quality validation and outlier detection to ensure accurate data input.
Training:
Mixed-precision training (AMP) to optimize memory usage.
Gradient clipping to avoid exploding gradients and ensure stable training.
OneCycleLR scheduling for efficient convergence.
Early stopping with configurable patience to avoid overfitting.
Checkpoint management for saving model, optimizer, and metrics during training.
Inference:
Batch processing for faster evaluation.
Morphological post-processing for mask refinement.
Visualization support with overlays and contour drawings.
Multi-image grid generation for displaying results across several test cases.
Documentation

This repository includes comprehensive documentation to guide you through:

Detailed architecture explanations.
Complete training instructions.
Inference and visualization guides.
Hyperparameter tuning strategies.
FAQ and troubleshooting section for resolving common issues.
Learning Outcomes
End-to-end autonomous driving perception pipeline using deep learning.
Best practices in deep learning: mixed precision, advanced training techniques.
Real-time model optimization methods suitable for edge devices.
Transfer learning and fine-tuning strategies for domain adaptation.
Multi-stage training methodologies that improve model performance across datasets.
File Overview
File	Purpose	Key Features
data_prep.py	HD map → image/mask generation	7-stage refinement, geometric projection
train.py	Base model training	Combined loss, AMP, OneCycleLR
finetune_bdd.py	Fine-tuning on BDD100K	Two-stage strategy, focal tversky loss
model.py	5-channel architecture	SE blocks, inverted residuals
model_rgb.py	3-channel architecture	Weight surgery, channel reduction
predict.py	Inference pipeline	Batch processing, visualization, FPS
dataset.py	nuScenes dataset loader	Augmentation pipeline with albumentations
dataset_bdd.py	BDD100K dataset loader	Automatic pairing, color label extraction
visualize.py	Ground truth display	Image-mask-overlay visualization
refine_masks.py	Post-hoc refinement	Morphology, texture, edge-based cleanup
Workflow
Raw nuScenes Dataset
        ↓
data_prep.py (geom projection + GrabCut + morphology)
        ↓
Refined Image-Mask Pairs
        ↓
train.py (nuScenes training)
        ↓
best_model.pth (5-channel checkpoint)
        ↓
finetune_bdd.py (weight surgery + BDD100K fine-tuning)
        ↓
best_model_bdd.pth ⭐ (0.8516 mIoU)
        ↓
predict.py (inference on test images)
        ↓
Predictions + Visualizations
Why This Project Stands Out
Production-Ready: Fully modular, documented, and tested codebase.
End-to-End Pipeline: From data collection to model training and inference.
State-of-the-Art Results: Achieves competitive accuracy (0.8516 mIoU) with much smaller models.
Real-Time Optimization: Efficient architecture with real-time deployment in mind.
Best Practices: Incorporates modern training techniques, transfer learning, and real-time optimizations.
References
MobileNetV3: https://arxiv.org/abs/1905.02244
Lite R-ASPP: https://arxiv.org/abs/2005.10910
Attention U-Net: https://arxiv.org/abs/1804.03999
nuScenes Dataset: https://www.nuscenes.org/
BDD100K Dataset: https://bdd100k.com/
Tags
autonomous-driving
semantic-segmentation
mobilenetv3
computer-vision
pytorch
real-time-inference
deep-learning
road-detection
bdd100k
nuscenes
embedded-ml
edge-computing
