# **Drivable Area Segmentation: Real-Time MobileNetV3 + LR-ASPP**

## **Overview**
This repository implements a high-performance real-time segmentation model for drivable area detection in autonomous driving. The model uses **MobileNetV3-Small** as a backbone, combined with the **Lite R-ASPP** decoder and **Attention U-Net** skip connections. The model is trained on the **nuScenes** dataset and fine-tuned on **BDD100K** for enhanced generalization. The segmentation model is optimized for real-time inference and supports both **RGB-only** and **RGB+LiDAR+Radar** input configurations.

---

## **Project Summary**

**Author**: Arindam Sushil Katoch  
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
```bash
pip install torch torchvision albumentations opencv-python tqdm nuscenes
