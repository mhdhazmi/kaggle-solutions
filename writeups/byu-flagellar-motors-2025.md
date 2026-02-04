# BYU - Locating Bacterial Flagellar Motors 2025

> Help locate flagellar motors in three-dimensional reconstructions of bacteria.

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | 3D Object Detection / Localization |
| **Data Domain** | Biomedical Imaging / Cryo-ET |
| **ML Approach** | 3D CNN + Detection Networks |
| **Key Techniques** | 3D Convolutions, U-Net, YOLO-style Detection |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $65,000 |
| **Teams** | 1,136 |
| **Timeline** | March - June 2025 |
| **Evaluation Metric** | F2-score + Euclidean Distance |
| **Host** | Brigham Young University |

## Problem Description

Develop an algorithm to identify the presence and location of flagellar motors in 3D tomographic reconstructions of bacteria.

### What is a Flagellar Motor?

The flagellar motor is a molecular machine that enables bacterial motility. It's essentially a biological rotary engine that spins the flagellum (bacterial tail), allowing bacteria to swim through liquids.

### The Challenge

- Detect whether a flagellar motor exists in a 3D tomogram
- If present, localize its precise 3D coordinates
- Handle low signal-to-noise ratio data
- Work with variable motor orientations
- Navigate crowded intracellular environments

### Why It Matters

- **Molecular Biology**: Understand fundamental cellular machinery
- **Drug Development**: Target bacterial motility for antibiotics
- **Synthetic Biology**: Engineer biological nanomachines
- **Research Acceleration**: Remove human bottleneck from cryo-ET analysis

---

## Data Description

### Dataset Structure

| Component | Description |
|-----------|-------------|
| `train/` | Subdirectories containing tomogram slices (JPEG stacks) |
| `train_labels.csv` | Motor locations for each tomogram |
| `test/` | ~900 tomograms for evaluation |

### Tomogram Structure

Each tomogram is a **3D volumetric image** provided as a stack of 2D JPEG slices:
- Each subdirectory = one tomogram
- Each JPEG = one slice (z-level)
- Reconstruct 3D volume by stacking slices

### Label Format

| Field | Description |
|-------|-------------|
| `tomo_id` | Unique tomogram identifier |
| `Motor axis 0` | Z-coordinate (slice number) |
| `Motor axis 1` | Y-coordinate |
| `Motor axis 2` | X-coordinate |
| `Array shape axis 0/1/2` | Tomogram dimensions |
| `Voxel spacing` | Angstroms per voxel (scale factor) |
| `Number of motors` | Count of motors in tomogram |

### Key Characteristics

- **Training data**: Multiple motors possible per tomogram
- **Test data**: Zero or one motor per tomogram
- **Prediction**: Set coordinates to -1 if no motor exists
- **Threshold**: Predictions within 1000 Angstroms count as correct

---

## Evaluation

**Metric**: F2-score combined with Euclidean distance

### Classification

- **True Positive (TP)**: `|y - ŷ|₂ ≤ 1000 Angstroms`
- **False Negative (FN)**: `|y - ŷ|₂ > 1000 Angstroms`
- **False Positive (FP)**: Predicting motor when none exists

### F2-Score Formula

```
F₂ = (1 + β²) × precision × recall / (β² × precision + recall)
```

Where β = 2 (recall weighted more than precision)

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $20,000 |
| 2nd | $15,000 |
| 3rd | $12,000 |
| 4th | $10,000 |
| 5th | $8,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- 3D U-Net architecture for detection
- Multi-scale feature extraction
- Ensemble of detection models
- Careful handling of class imbalance

### 2nd Place Solution

**Technique**:
- 3D convolutional neural networks
- Anchor-based detection similar to YOLO
- Heavy data augmentation (rotations, flips)

### 3rd Place Solution

**Innovation**:
- Transformer-based 3D detection
- Cross-attention between slices
- Template matching refinement

---

## Common Winning Strategies

### 1. 3D Architecture Design

| Architecture | Use Case |
|--------------|----------|
| 3D U-Net | Segmentation + localization |
| 3D ResNet | Feature extraction backbone |
| V-Net | Medical imaging standard |
| YOLO3D | Direct detection |

### 2. Data Preprocessing

```
1. Load JPEG stack
2. Stack into 3D numpy array
3. Normalize intensities
4. Apply contrast enhancement (CLAHE)
5. Resize/pad to consistent dimensions
```

### 3. Handling Low SNR

- Denoising with learned filters
- Multi-slice context aggregation
- Averaging predictions from multiple views

### 4. Augmentation Strategies

- 3D rotations (especially important for varying motor orientations)
- 3D flips along all axes
- Intensity perturbations
- Elastic deformations

### 5. Two-Stage Pipeline

```
Stage 1: Binary classification (motor present?)
Stage 2: If positive, localize motor center
```

---

## Technical Insights

### Understanding Tomograms

A tomogram is reconstructed from a tilt series:
- Electron microscope captures 2D projections at different angles
- Back-projection algorithms reconstruct 3D volume
- Result: Low SNR but preserved molecular structure

### Motor Detection Challenges

1. **Variable orientation**: Motors can point any direction
2. **Crowded environment**: Many other cellular structures present
3. **Low contrast**: Motor may be barely visible
4. **Size variation**: Voxel spacing affects apparent size

### Coordinate System

- Axis 0 (Z): Slice number (depth)
- Axis 1 (Y): Height in slice
- Axis 2 (X): Width in slice

### Voxel Spacing Importance

The `voxel_spacing` field indicates real-world scale:
- Higher spacing = lower resolution
- Must consider when applying 1000 Å threshold
- Normalize coordinates for consistent detection

---

## Code Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Notebook | ≤12 hours |
| GPU Notebook | ≤12 hours |
| Internet | Disabled |
| Submission | `submission.csv` |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583143) |
| 2nd | [Discussion](https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/584980) |
| 3rd | [Discussion](https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583380) |
| 4th | [Discussion](https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583411) |
| 6th | [Discussion](https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/587410) |

---

## Citation

```bibtex
@misc{byu-flagellar-motors-2025,
    title = {BYU - Locating Bacterial Flagellar Motors 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025}},
    note = {Brigham Young University, Kaggle}
}
```
