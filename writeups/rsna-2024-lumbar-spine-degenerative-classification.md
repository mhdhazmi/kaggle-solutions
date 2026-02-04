# RSNA 2024 Lumbar Spine Degenerative Classification

> Classify lumbar spine degenerative conditions

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Label Multi-Class Classification |
| **Data Domain** | Medical Imaging / Radiology |
| **ML Approach** | Deep Learning (CNN/Transformer) |
| **Key Techniques** | 3D Medical Image Analysis, Multi-Task Learning |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,874 |
| **Timeline** | May - October 2024 |
| **Evaluation Metric** | RSNA Lumbar Metric (Weighted Log Loss) |
| **Host** | RSNA + ASNR |

## Problem Description

Create models to detect and classify degenerative spine conditions using lumbar spine MR images, simulating a radiologist's diagnostic performance.

### The Challenge

- Classify 5 degenerative conditions
- Across 5 intervertebral disc levels (L1/L2 to L5/S1)
- 3 severity levels per condition
- Total: 25 classification tasks per study

### Why It Matters

- **Low Back Pain**: Leading cause of disability worldwide (619M people in 2020)
- **Early Detection**: Proper diagnosis guides treatment decisions
- **AI Assistance**: Aid radiologists in consistent grading
- **Patient Outcomes**: Better diagnosis improves quality of life

---

## Data Description

### Conditions Classified

| Condition | Description |
|-----------|-------------|
| Left Neural Foraminal Narrowing | Narrowing of left nerve exit |
| Right Neural Foraminal Narrowing | Narrowing of right nerve exit |
| Left Subarticular Stenosis | Compression in left lateral recess |
| Right Subarticular Stenosis | Compression in right lateral recess |
| Spinal Canal Stenosis | Central canal narrowing |

### Severity Levels

| Level | Description |
|-------|-------------|
| Normal/Mild | No significant abnormality |
| Moderate | Moderate degeneration |
| Severe | Severe degeneration |

### Vertebral Levels

- L1/L2, L2/L3, L3/L4, L4/L5, L5/S1

### Files

| File | Description |
|------|-------------|
| `train.csv` | Labels with study_id and condition severities |
| `train_label_coordinates.csv` | Coordinates marking relevant areas |
| `train_images/` | DICOM MRI images |
| `sample_submission.csv` | Submission format |

### Ground Truth

Labels created by spine radiology specialists with:
- Axial T2
- Sagittal T1
- Sagittal T2/STIR sequences

---

## Evaluation

**Metric**: RSNA Lumbar Metric (Weighted Log Loss)

```python
# For each condition and level
# Predict probabilities for Normal/Mild, Moderate, Severe
# Weighted by sample weights (emphasizes severe cases)
```

The metric uses sample weights to handle class imbalance, giving more importance to correctly classifying severe conditions.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Author**: NANACHI

**Key Approach**:
- Multi-stage pipeline
- Anatomical localization first
- Level-specific classification
- Ensemble of multiple architectures

### 2nd Place Solution

**Author**: YujiAriyasu

**Technique**:
- EfficientNet-based backbone
- 3D convolutions for slice context
- Multi-task learning
- Heavy augmentation

### 4th Place Solution

**Author**: tattaka

**Innovation**:
- Transformer architecture
- Attention to relevant slices
- Cross-attention between sequences
- Efficient training strategy

### 5th Place Solution

**Author**: siwooyong

**Approach**:
- CNN backbone with LSTM
- Sequence modeling across slices
- Multi-view fusion

---

## Common Winning Strategies

### 1. Two-Stage Pipeline

Most winning solutions used:
1. **Stage 1**: Localize vertebral levels and relevant anatomy
2. **Stage 2**: Classify conditions at each level

```python
# Stage 1: Localization
level_detector = KeypointModel()
level_coords = level_detector(mri_volume)

# Stage 2: Classification per level
for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
    roi = extract_roi(mri_volume, level_coords[level])
    predictions[level] = classifier(roi)
```

### 2. Multi-Sequence Fusion

```python
class MultiSequenceModel(nn.Module):
    def __init__(self):
        self.axial_encoder = Encoder3D()
        self.sagittal_t1_encoder = Encoder3D()
        self.sagittal_t2_encoder = Encoder3D()
        self.fusion = AttentionFusion()

    def forward(self, axial, sag_t1, sag_t2):
        axial_feat = self.axial_encoder(axial)
        sag_t1_feat = self.sagittal_t1_encoder(sag_t1)
        sag_t2_feat = self.sagittal_t2_encoder(sag_t2)
        fused = self.fusion(axial_feat, sag_t1_feat, sag_t2_feat)
        return self.classifier(fused)
```

### 3. 3D Medical Image Processing

```python
import nibabel as nib
import pydicom

def load_mri_series(dicom_folder):
    slices = []
    for f in sorted(os.listdir(dicom_folder)):
        ds = pydicom.dcmread(os.path.join(dicom_folder, f))
        slices.append(ds.pixel_array)

    volume = np.stack(slices, axis=0)
    return volume

# Preprocessing
volume = normalize_intensity(volume)
volume = resize_volume(volume, target_shape=(32, 256, 256))
```

### 4. Multi-Task Learning

```python
class SpineClassifier(nn.Module):
    def __init__(self):
        self.backbone = EfficientNet3D()
        self.heads = nn.ModuleDict({
            'spinal_canal_stenosis': ClassificationHead(3),
            'left_neural_foraminal': ClassificationHead(3),
            'right_neural_foraminal': ClassificationHead(3),
            'left_subarticular': ClassificationHead(3),
            'right_subarticular': ClassificationHead(3),
        })

    def forward(self, x):
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}
```

### 5. Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV3D

augmentations = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ElasticTransform(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
])
```

---

## Technical Insights

### MRI Sequences

| Sequence | Purpose |
|----------|---------|
| Axial T2 | Cross-sectional view, disc/stenosis visualization |
| Sagittal T1 | Longitudinal view, anatomical detail |
| Sagittal T2/STIR | Fluid-sensitive, pathology highlight |

### Anatomical Understanding

Key structures:
- **Vertebral bodies**: L1-L5 and S1
- **Intervertebral discs**: Between vertebrae
- **Neural foramina**: Where nerves exit
- **Spinal canal**: Contains spinal cord
- **Subarticular zone**: Lateral recess area

### Class Imbalance

| Severity | Typical Distribution |
|----------|---------------------|
| Normal/Mild | ~70% |
| Moderate | ~20% |
| Severe | ~10% |

Handled via:
- Sample weights in loss
- Oversampling severe cases
- Focal loss

### Siamese Network Approach

One notable approach used Siamese networks to compare left vs right sides for bilateral conditions.

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | 9 hours |
| Internet | Disabled |
| GPU | Available |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091) |
| 2nd | [Discussion](https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452) |
| 3rd | [Discussion](https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453) |
| 4th | [Discussion](https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539443) |
| 5th | [Discussion](https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472) |

---

## Citation

```bibtex
@misc{rsna-2024-lumbar-spine,
    author = {RSNA and ASNR},
    title = {RSNA 2024 Lumbar Spine Degenerative Classification},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification}},
    note = {Kaggle}
}
```
