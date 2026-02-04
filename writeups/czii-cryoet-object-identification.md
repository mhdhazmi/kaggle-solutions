# CZII - CryoET Object Identification

> Find small biological structures in large 3D volumes

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | 3D Object Detection / Instance Segmentation |
| **Data Domain** | Biomedical Imaging / Cryo-Electron Tomography |
| **ML Approach** | 3D CNN + Detection Networks |
| **Key Techniques** | 3D U-Net, Point Detection, Template Matching |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $75,000 |
| **Teams** | 931 |
| **Timeline** | 2024-2025 |
| **Evaluation Metric** | Custom CZI CryoET Metric |
| **Host** | Chan Zuckerberg Initiative |

## Problem Description

Identify and localize small biological structures (molecular complexes) in large 3D cryo-electron tomography volumes.

### The Challenge

- Detect multiple types of biological structures
- Handle extremely large 3D volumes
- Work with low signal-to-noise ratio data
- Find structures at various scales and orientations

### Why It Matters

- **Structural Biology**: Understand molecular machinery
- **Drug Discovery**: Target identification and validation
- **Basic Science**: Cellular organization research
- **Automation**: Scale up structural biology workflows

---

## Data Description

### What is Cryo-ET?

Cryo-electron tomography (cryo-ET) is an imaging technique that:
1. Flash-freezes biological samples
2. Captures images at multiple tilt angles
3. Reconstructs 3D tomograms
4. Preserves native molecular structures

### Dataset Structure

| Component | Description |
|-----------|-------------|
| Tomograms | Large 3D volumes (hundreds of GB) |
| Annotations | Point locations of structures |
| Structure types | Multiple particle classes |

### Key Characteristics

- **Volume size**: Very large 3D arrays
- **Low SNR**: Noisy due to radiation damage limits
- **Multiple targets**: Several structure types to detect
- **Varying orientations**: Structures in arbitrary poses

---

## Evaluation

**Metric**: Custom CZI CryoET Metric

Evaluates:
- Detection accuracy (precision/recall)
- Localization accuracy
- Per-class performance
- Overall F-score variants

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |

**Total Prize Pool**: $75,000

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- 3D U-Net for semantic segmentation
- Post-processing to extract point locations
- Multi-scale feature extraction
- Heavy data augmentation

### 2nd Place Solution

**Technique**:
- Point-based detection network
- Attention mechanisms for context
- Ensemble of detection models

### 3rd Place Solution

**Innovation**:
- Template matching combined with CNN
- Learned feature representations
- Adaptive thresholding

---

## Common Winning Strategies

### 1. Architecture Design

| Architecture | Use Case |
|--------------|----------|
| 3D U-Net | Semantic segmentation |
| PointNet variants | Direct point prediction |
| 3D ResNet | Feature backbone |
| Detection heads | CenterNet-style |

### 2. Handling Large Volumes

```python
# Sliding window inference
def process_large_volume(volume, model, patch_size, overlap):
    predictions = []
    for patch, coords in sliding_window(volume, patch_size, overlap):
        pred = model(patch)
        predictions.append((pred, coords))
    return merge_predictions(predictions)
```

### 3. Data Augmentation

Critical for limited training data:
- 3D rotations (all axes)
- 3D flips
- Intensity augmentation
- Elastic deformation
- Noise injection

### 4. Multi-Class Detection

Handle different structure types:
- Separate detection heads per class
- Class-balanced sampling
- Different thresholds per class

### 5. Post-Processing

Convert predictions to point locations:
- Non-maximum suppression (3D)
- Connected component analysis
- Peak detection
- Confidence thresholding

---

## Technical Insights

### Signal-to-Noise Challenges

Cryo-ET data has inherently low SNR:
- Limited electron dose (radiation damage)
- Missing wedge artifacts
- Ice thickness variations

Mitigation strategies:
- Denoising networks
- Multi-scale analysis
- Ensemble averaging

### Memory Management

Large 3D volumes require careful handling:
```python
# Use memory-mapped files
tomogram = np.memmap(path, dtype='float32', mode='r', shape=shape)

# Process in patches
for patch in generate_patches(tomogram, patch_size):
    process(patch)
```

### Structure Variability

Biological structures vary in:
- Size (different molecular complexes)
- Orientation (random in tomogram)
- Conformation (structural flexibility)
- Context (membrane-bound vs cytoplasmic)

### Evaluation Nuances

- Distance threshold for "correct" detection
- Handling of closely spaced particles
- Class imbalance between structure types

---

## Code Requirements

| Requirement | Limit |
|-------------|-------|
| GPU Runtime | Extended time allowed |
| Memory | High memory instances |
| Internet | Disabled |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561510) |
| 2nd | [Discussion](https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561568) |
| 3rd | [Discussion](https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561417) |
| 4th | [Discussion](https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561401) |
| 5th | [Discussion](https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561580) |

---

## Citation

```bibtex
@misc{czii-cryoet-object-identification,
    title = {CZII - CryoET Object Identification},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/czii-cryo-et-object-identification}},
    note = {Chan Zuckerberg Initiative, Kaggle}
}
```
