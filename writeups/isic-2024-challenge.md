# ISIC 2024 - Skin Cancer Detection with 3D-TBP

> Identify cancers among skin lesions cropped from 3D total body photographs

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Medical Imaging / Dermatology |
| **ML Approach** | Deep Learning (CNN + Metadata) |
| **Key Techniques** | Image Classification, Tabular Features, Ensemble |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $80,000 |
| **Teams** | 2,739 |
| **Timeline** | June - September 2024 |
| **Evaluation Metric** | pAUC above 80% TPR |
| **Host** | ISIC (International Skin Imaging Collaboration) |

## Problem Description

Develop algorithms to identify histologically confirmed skin cancer from single-lesion crops extracted from 3D total body photographs, mimicking non-dermoscopic images like smartphone photos.

### The Challenge

- Binary classification: malignant vs benign
- Non-dermoscopic images (smartphone-like quality)
- Extreme class imbalance (few cancers)
- Must achieve high sensitivity (TPR > 80%)

### Why It Matters

- **Early Detection**: Key factor in long-term patient outcomes
- **Accessibility**: Algorithms for settings without dermatologists
- **Triage**: Help decide who needs specialist care
- **Global Impact**: Skin cancer affects millions worldwide

---

## Data Description

### The SLICE-3D Dataset

Standardized cropped lesion images from 3D Total Body Photography:
- **Vectra WB360**: Captures complete visible skin surface
- **AI-based lesion detection**: Automatically identifies individual lesions
- **Non-dermoscopic quality**: Resembles cell phone photos

### Image Characteristics

| Property | Description |
|----------|-------------|
| Format | JPEG images |
| Quality | Macro-resolution, non-dermoscopic |
| Source | 3D TBP cropped lesions |
| Challenge | Lower quality than dermoscopy |

### Target

Binary classification:
- `target = 0`: Benign lesion
- `target = 1`: Malignant (histologically confirmed)

### Metadata Features

| Feature | Description |
|---------|-------------|
| age_approx | Approximate patient age |
| sex | Patient sex |
| anatom_site_general | Body location |
| image source | Origin of image |
| precise diagnosis | Detailed diagnosis |

### Class Imbalance

Extreme imbalance - malignant cases are rare:
- ~99% benign
- ~1% malignant

---

## Evaluation

**Metric**: Partial AUC above 80% TPR (pAUC80)

```python
# Only count TPR >= 0.80 region of ROC curve
# Normalized to [0, 1] range

from sklearn.metrics import roc_curve, auc

def pauc_above_tpr(y_true, y_score, min_tpr=0.8):
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Filter to TPR >= min_tpr
    mask = tpr >= min_tpr
    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]

    # Calculate partial AUC
    partial_auc = auc(fpr_partial, tpr_partial)

    # Normalize
    max_area = (1 - min_tpr) * 1.0  # Max possible area
    return partial_auc / max_area
```

### Why pAUC80?

For cancer screening:
- **Must catch cancers** (high sensitivity)
- **Specificity matters too** (avoid false alarms)
- pAUC80 rewards models that maintain good specificity while ensuring 80%+ sensitivity

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $80,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Ensemble of image models + tabular
- Heavy augmentation
- Multi-resolution inputs
- Pseudo-labeling

### 2nd Place Solution

**Technique**:
- EfficientNet variants
- Metadata integration
- Custom loss for pAUC optimization
- TTA (Test Time Augmentation)

### 3rd Place Solution

**Innovation**:
- Vision Transformer backbone
- Patient-level aggregation
- Careful handling of imbalance

---

## Common Winning Strategies

### 1. Image Model Backbone

```python
import timm

# Common backbones
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1)

# Or Vision Transformer
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)

# Or ConvNeXt
model = timm.create_model('convnext_base', pretrained=True, num_classes=1)
```

### 2. Handling Extreme Class Imbalance

```python
# Weighted loss
pos_weight = torch.tensor([n_neg / n_pos])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Oversampling malignant cases
from torch.utils.data import WeightedRandomSampler
weights = [1.0 if label == 0 else 100.0 for label in labels]
sampler = WeightedRandomSampler(weights, len(weights))
```

### 3. Metadata Integration

```python
class ImageTabularModel(nn.Module):
    def __init__(self, image_model, n_tabular_features):
        super().__init__()
        self.image_encoder = image_model
        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_features + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_encoder(tabular)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(combined)
```

### 4. Data Augmentation

```python
import albumentations as A

train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(),
    ToTensorV2(),
])
```

### 5. Test Time Augmentation (TTA)

```python
def predict_with_tta(model, image, n_augmentations=5):
    predictions = []

    # Original
    predictions.append(model(image))

    # Horizontal flip
    predictions.append(model(torch.flip(image, dims=[3])))

    # Vertical flip
    predictions.append(model(torch.flip(image, dims=[2])))

    # Multiple random augmentations
    for _ in range(n_augmentations - 3):
        aug_image = apply_random_augmentation(image)
        predictions.append(model(aug_image))

    return torch.stack(predictions).mean(dim=0)
```

---

## Technical Insights

### Non-Dermoscopic vs Dermoscopic

| Aspect | Dermoscopy | Non-Dermoscopic |
|--------|------------|-----------------|
| Magnification | 10-20x | None |
| Subsurface | Visible | Not visible |
| Quality | Consistent | Variable |
| Accessibility | Specialist | Anyone |

### pAUC Optimization

Direct optimization approaches:
```python
# Surrogate loss for pAUC
def pauc_loss(y_pred, y_true, min_tpr=0.8):
    # Sort by prediction
    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]

    # Find threshold for min_tpr
    n_pos = y_true.sum()
    tpr_threshold = int(n_pos * min_tpr)

    # Calculate loss focusing on relevant region
    # ... custom implementation
```

### Melanoma Characteristics

Visual features AI should learn:
- **Asymmetry**: Irregular shape
- **Border**: Uneven edges
- **Color**: Multiple colors
- **Diameter**: > 6mm
- **Evolution**: Changes over time (not available in static images)

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
| 1st | [Discussion](https://www.kaggle.com/c/isic-2024-challenge/discussion/533196) |
| 2nd | [Discussion](https://www.kaggle.com/c/isic-2024-challenge/discussion/532704) |
| 3rd | [Discussion](https://www.kaggle.com/c/isic-2024-challenge/discussion/532919) |
| 4th | [Discussion](https://www.kaggle.com/c/isic-2024-challenge/discussion/532760) |
| 5th | [Discussion](https://www.kaggle.com/c/isic-2024-challenge/discussion/533056) |

---

## Citation

```bibtex
@misc{isic-2024-challenge,
    author = {ISIC},
    title = {ISIC 2024 - Skin Cancer Detection with 3D-TBP},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/isic-2024-challenge}},
    note = {Kaggle}
}
```
