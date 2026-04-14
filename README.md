# **Drivable Area Segmentation: Real-Time MobileNetV3 + LR-ASPP**

## **Overview**
This repository implements a high-performance real-time segmentation model for drivable area detection in autonomous driving. The model uses **MobileNetV3-Small** as a backbone, combined with the **Lite R-ASPP** decoder and **Attention U-Net** skip connections. The model is trained on the **nuScenes** dataset and fine-tuned on **BDD100K** for enhanced generalization. The segmentation model is optimized for real-time inference and supports both **RGB-only** and **RGB+LiDAR+Radar** input configurations.

---

## **Project Summary**

**Author**: Anish Kapila
**Challenge**: MAHE-MOBILITY CHALLENGE/2026 (Team Submission)  
**Role**: Full-stack development (model architecture, training pipeline, data preprocessing, evaluation)

### **Key Features**
- **Model Size**: ~1.1 million parameters
- **Inference Speed**: ~70 FPS on GPU, optimized for embedded systems
- **Training Pipeline**: Multi-stage pipeline (nuScenes → BDD100K)
- **Loss Functions**: Weighted BCE, Dice, Sobel boundary loss
- **Multi-sensor Fusion**: RGB-only and multi-channel (RGB+LiDAR+Radar)
- **Robust Data Pipeline**: 7-stage mask generation with morphological refinement

---

## **File Overview**

### **Data Preparation and Post-Processing**

- **`data_prep.py`**  
  Generates image-mask pairs from **nuScenes HD maps** for model training. It preprocesses the data to create suitable inputs for training.
  
- **`refine_masks.py`**  
  Post-processes the generated masks to refine them using **morphological operations**, improving accuracy and segmentation boundary precision.

- **`verify_masks.py`**  
  Verifies the integrity of the generated masks, ensuring that they align correctly with the ground truth, and checks for potential errors or inconsistencies.

- **`visualize.py`**  
  Provides visualizations of ground truth and predicted masks. Useful for debugging and assessing model performance.

### **Model Architecture and Training**

- **`model_rgb.py`**  
  Defines the segmentation model architecture, optimized for **RGB-only input**. This model is a variant of the full model optimized for real-time use with fewer input channels.

- **`train.py`**  
  Implements the training pipeline for the base model on the **nuScenes** dataset. It includes mixed-precision training (AMP) and other optimization techniques for efficient training.

- **`finetune_bdd.py`**  
  Fine-tunes the model on the **BDD100K** dataset to improve generalization. This script builds on the pretrained model and refines it for better performance on the BDD100K dataset.

- **`predict.py`**  
  Runs inference on the trained/fine-tuned model, generating predictions for test images. Includes visualization of the output masks for verification.

### **Miscellaneous**

- **`finetune_curves.png`**  
  A plot of the training and validation loss curves during the fine-tuning process, helpful for visualizing model performance during training.

- **`best_model_bdd.pth`**  
  The final fine-tuned model after training on **BDD100K**, achieving an impressive **0.8516 mIoU** on the validation set.

---

## **Quick Start Guide**

### **Installation**

Install the necessary dependencies:

pip install torch torchvision albumentations opencv-python tqdm nuscenes

Data Preparation

Prepare the nuScenes data:

python data_prep.py
Model Training

Train the base model on nuScenes:

python train.py
Fine-Tuning on BDD100K

Fine-tune the model on BDD100K:

python finetune_bdd.py
Inference

Run inference on test images:

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
Geometric Projection: Converts 3D HD map polygons to 2D camera images for better segmentation accuracy.
GrabCut Refinement: Uses projected masks as seeds for interactive segmentation.
Boundary Loss: Implements Sobel-based loss for enhancing road boundary accuracy.
Weight Surgery: Reduces a 5-channel model to a 3-channel model without performance degradation.
Two-Stage Fine-Tuning: Initial warm-up with frozen encoder, followed by full model training.
Model Architecture

The model utilizes an encoder-decoder architecture for semantic segmentation:

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
Autonomous Vehicle Navigation: Helps autonomous vehicles determine safe and drivable paths.
Drivable Surface Detection: Detects roads, parking lots, driveways, and more.
Scene Understanding: Assists vehicles and robots in understanding their environment.
Mobile Robotics: Enables robots to avoid obstacles and navigate efficiently.
Drone Landing Zones: Identifies safe landing zones for autonomous drones.
Advanced Features
Data Pipeline: Efficient data processing pipeline, including geometric transformations and mask refinement.
Training: Uses mixed-precision, gradient clipping, and OneCycleLR scheduling for faster and more stable training.
Inference: Supports batch processing, morphological post-processing, and visualizations for real-time results.
Documentation

This repository includes detailed documentation for:

Full model architecture
End-to-end training process
Inference and visualization instructions
Hyperparameter tuning tips
Troubleshooting guides
Learning Outcomes

Learn how to build an end-to-end autonomous driving perception pipeline.
Gain experience with modern deep learning techniques: mixed precision, scheduling, and transfer learning.
Understand how to deploy a real-time, efficient deep learning model on embedded systems.
Learn the details of multi-stage training pipelines and data processing techniques.

Why This Project Stands Out
Modular and Production-Ready: Well-structured, documented codebase for easy integration.
Competitive Performance: 0.8516 mIoU with a lightweight model (1.1M parameters).
Real-Time Performance: Optimized for embedded systems and edge devices.
Transfer Learning: Effective fine-tuning from nuScenes to BDD100K dataset.
Best Practices: Implements modern deep learning techniques like AMP, OneCycleLR, and gradient clipping.

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


