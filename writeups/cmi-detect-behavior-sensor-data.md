# CMI - Detect Behavior with Sensor Data

> Classify hand gestures and behaviors from smartwatch sensor data

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-class Classification / Time Series |
| **Data Domain** | Sensor Data / Wearable Devices / IMU |
| **ML Approach** | Deep Learning (1D-CNN, Transformers, 2D-CNN) |
| **Key Techniques** | Sensor Fusion, Hungarian Assignment, Online Pseudo-Labeling |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Teams** | ~2,000+ |
| **Timeline** | 2025 |
| **Evaluation Metric** | Accuracy |
| **Host** | CMI (Child Mind Institute) |

## Problem Description

Classify hand gestures and orientation from smartwatch sensor data. The competition involves predicting combinations of **18 gestures** × **4 orientations** = **72 possible combinations** (with 51 actually present in data).

### The Task

Given time-series sensor data from a smartwatch, predict:
- **Gesture**: Which of 18 hand gestures is being performed
- **Orientation**: The device orientation during the gesture

### Data Modalities

1. **IMU (Inertial Measurement Unit)**: Accelerometer + Gyroscope + Quaternion rotation
2. **THM (Thermal)**: 5-channel thermal sensor data
3. **ToF (Time of Flight)**: 8×8 depth images

### Key Challenge

- **Online prediction**: Predictions made one at a time via API
- **Subject-specific patterns**: Each subject has exactly 51 (gesture, orientation) pairs × 2 behaviors = 102 sequences
- **No-repeat constraint**: Each class appears at most twice per subject

---

## Data Description

### Sensor Channels

| Sensor | Channels | Description |
|--------|----------|-------------|
| **Accelerometer** | 3 (x, y, z) | Linear acceleration |
| **Gyroscope** | 3 (x, y, z) | Angular velocity |
| **Quaternion** | 4 (w, x, y, z) | Device rotation |
| **Thermal (THM)** | 5 channels | Thermal sensor readings |
| **Time of Flight (ToF)** | 8×8 depth | Distance measurements |

### Dataset Structure

- **Training**: ~8k sequences with labels
- **Test**: ~3.5k sequences (online prediction via API)
- **Per subject**: 102 sequences (51 gesture-orientation pairs × 2 behaviors)

### Known Data Issues

Two subjects had **inverted device orientation**:
- `SUBJ_019262`
- `SUBJ_045235`

Required correction: 180° rotation around z-axis for quaternion and ToF data, plus sign swaps for accelerometer.

---

## Evaluation

**Metric**: Classification Accuracy

**Online Prediction**: Predictions submitted one at a time via API (cannot revise previous predictions).

---

## Top Solutions

### 1st Place Solution

**Author**: yuanzhe zhou

*(Details to be added)*

---

### 2nd Place Solution (Score: 0.878 Private LB)

**Author**: daiwakun

#### Architecture: Hierarchical 1D-CNN with Attention

```
Input → Feature Extraction → 1D CNN Blocks → Phase-Aware Attention → Classification
```

**Key Components**:
- **Phase-aware attention**: Auxiliary phase predictor for long vs. short phases
- **Positional encoding**: Time step information
- **Ensemble**: 3 variants with different depths/layers

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | Cosine Annealing |
| Batch size | 32 |
| Epochs | 50 |
| Folds | 10 |

#### Revolutionary Post-Processing: Hungarian Assignment

**Key Insight**: Dataset structure guarantees each subject has exactly 51 unique (gesture, orientation) pairs, each appearing twice.

**Approach**:
- Predict 102 classes (composite label: gesture × orientation × initial_behavior)
- For each subject, maintain cache of all predicted probabilities
- At each new prediction, re-optimize joint assignment using **Hungarian algorithm**
- Maximize sum of log-probabilities under no-repeat constraint (each class used at most twice)

**Online Pseudo-Labeling**:
- Accumulate test sequences in batches
- Fine-tune with small LR (5e-5) using pseudo labels

#### Results

| Method | Public LB | Private LB |
|--------|-----------|------------|
| No pseudo / No post-proc | 0.862 | 0.854 |
| With pseudo / No post-proc | 0.865 | 0.858 |
| No pseudo / With post-proc | 0.891 | 0.875 |
| **With pseudo / With post-proc** | **0.900** | **0.878** |

---

### 3rd Place Solution (Score: 0.878 Private LB)

**Team**: theoviel, ren4yu, nazarov, minerppdy

#### Multi-Pipeline Ensemble

**IMU-only branch** + **Full-data branch** (IMU + THM + ToF)

#### Shared Key Techniques

**1. Handedness Normalization**
For samples with `handedness = 1`, canonicalize to common side:
- IMU: Left-right flip using quaternion rotation (-110° on z-axis)
- THM: Swap 3rd and 5th channels
- ToF: Swap 3rd and 5th channels + image flip

**2. Quaternion Handling**
- Convention: Keep `rot_w` positive
- Sign flip augmentation or symmetric blocks: `rot_block(quat) + rot_block(-quat)`
- Sequence smoothing for discontinuities

**3. Data Augmentation**
- Mixup
- ToF dropout
- Stretch & shift sequence
- Quaternion rotation augmentation
- Cutmix of sequences

**4. Subject-Specific Corrections**
Fixed `SUBJ_019262` and `SUBJ_045235` device orientation issues.

#### Individual Pipelines

**Theo's Pipeline (Transformers + CNNs)**:
- SE-1D-CNN blocks with different kernel sizes
- GRU, DeBERTa, Squeezeformer variants
- Gesture/transition mask head for pooling
- Best backbones: MaxViT, ConvNeXt-v2
- IMU CV: 0.844, Full CV: 0.897

**Yu4u's Pipeline**:
- EfficientNet-like 1D CNN backbone
- ModernBERT backbone for IMU-only
- Auxiliary heads for behavior and orientation
- IMU CV: 0.8423, Full CV: 0.8857

**Leonid's Pipeline (2D CNNs)**:
- Novel approach: Treat (n_timestamps, n_features) as image
- Resize to (image_size, image_size), repeat 3× for RGB
- EfficientNet family (B0, B3, B5, V2_S, V2_M)
- Features separated by zeros in image
- IMU CV: 0.8310, Full CV: 0.8868

**Minerppdy's Pipeline**:
- Multi-branch architecture for different input channels
- Relative quaternion to next step
- Angular jerk/snap, Acc jerk/snap features
- IMU CV: 0.8283, Full CV: 0.8850

#### Post-Processing

Same Hungarian assignment approach as 2nd place:
- Limit each subject to 2 predictions per (gesture, orientation)
- Re-optimize from scratch for each new sample
- +1.5% improvement (0.882 → 0.897 Public)

#### Final Results

| Configuration | Public LB | Private LB |
|---------------|-----------|------------|
| Base ensemble | 0.882 | 0.864 |
| **With Hungarian PP** | **0.897** | **0.878** |

---

## Common Winning Strategies

### 1. Handedness Normalization
Canonicalizing left/right handed samples to consistent orientation.

### 2. Hungarian Assignment Post-Processing
Leveraging dataset structure (102 sequences per subject, no repeats) for global optimization.

### 3. Multi-Modal Sensor Fusion
Combining IMU, thermal, and ToF data with specialized branches.

### 4. 2D CNN on Time Series
Treating (time × features) as images for pre-trained CNN backbones.

### 5. Quaternion-Aware Augmentation
Physics-informed augmentation respecting quaternion properties.

### 6. Online Pseudo-Labeling
Incrementally improving models with test-time predictions.

---

## Technical Insights

### Why Hungarian Assignment Works

1. **Structured dataset**: Exactly 51 unique (gesture, orientation) pairs per subject
2. **No-repeat constraint**: Each class appears at most twice
3. **Joint optimization**: Re-evaluating all predictions together beats greedy selection
4. **Online adaptation**: Can recover from early mistakes by reassigning

### Key Data Preprocessing

1. **Device orientation correction** for specific subjects
2. **Quaternion sign convention** (positive w)
3. **Padding/truncating on left** (more signal at end of gesture)

---

## Technical Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | ≤12 hours |
| GPU Runtime | ≤12 hours |
| Internet Access | Disabled |
| External Data | Allowed |

---

## Solution Links

| Place | Team | Score | Solution |
|-------|------|-------|----------|
| 1st | yuanzhe zhou | - | [Writeup](https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution) |
| 2nd | daiwakun | 0.878 | [Writeup](https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution) |
| 3rd | theoviel et al. | 0.878 | [Writeup](https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution) |

---

## Citation

```bibtex
@misc{cmi-detect-behavior-sensor-data,
    title = {CMI - Detect Behavior with Sensor Data},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/cmi-detect-behavior-with-sensor-data}},
    note = {Kaggle}
}
```
