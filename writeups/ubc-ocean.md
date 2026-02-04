# UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)

> Navigating Ovarian Cancer: Unveiling Common Histotypes and Unearthing Rare Variants

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Class Image Classification + Outlier Detection |
| **Data Domain** | Medical Imaging / Histopathology |
| **ML Approach** | Deep Learning (CNNs, Vision Transformers) |
| **Key Techniques** | Whole Slide Image Processing, Transfer Learning, Multi-Instance Learning |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,326 |
| **Timeline** | October 2023 - January 2024 |
| **Evaluation Metric** | Balanced Accuracy Score |
| **Host** | University of British Columbia |

## Problem Description

Classify ovarian cancer subtypes from histopathology images using the world's most extensive ovarian cancer dataset from 20+ medical centers.

### The Challenge

- Classify 5 common ovarian cancer subtypes
- Detect rare variants (outliers) not seen in training
- Handle massive whole slide images (WSI)
- Generalize across different medical centers

### The 6 Classes

| Code | Subtype | Description |
|------|---------|-------------|
| HGSC | High-Grade Serous Carcinoma | Most common (~70%) |
| CC | Clear Cell Carcinoma | Distinct cellular morphology |
| EC | Endometrioid Carcinoma | Endometrium-like cells |
| LGSC | Low-Grade Serous Carcinoma | Less aggressive |
| MC | Mucinous Carcinoma | Mucin-producing cells |
| Other | Rare Variants (Outliers) | Not in training set |

### Why It Matters

- **Diagnosis**: Improve accuracy and reduce inter-observer variability
- **Access**: Enable diagnosis in underserved communities
- **Treatment**: Subtype-specific therapies require accurate identification
- **Research**: Better understanding of ovarian cancer heterogeneity

---

## Data Description

### Image Types

| Type | Description | Size |
|------|-------------|------|
| WSI | Whole Slide Images | Up to 100,000 x 50,000 pixels |
| TMA | Tissue Microarray | ~4,000 x 4,000 pixels |

### Files

| File | Description |
|------|-------------|
| `train_images/` | Training WSI and TMA images |
| `test_images/` | Test images (~2,000, 550GB total) |
| `train_thumbnails/` | Smaller versions of WSIs |
| `train.csv` | Labels and metadata |
| `supplemental_masks/` | Cancer/healthy/necrotic region masks |

### Data Properties

| Property | Value |
|----------|-------|
| WSI Magnification | 20x |
| TMA Magnification | 40x |
| Training Centers | 20+ worldwide |
| Test Centers | Different from training |

---

## Evaluation

**Metric**: Balanced Accuracy Score

```python
from sklearn.metrics import balanced_accuracy_score

def evaluate(y_true, y_pred):
    """
    Balanced accuracy: average recall per class
    Handles class imbalance by weighting equally
    """
    return balanced_accuracy_score(y_true, y_pred)
```

Higher is better. Accounts for class imbalance.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $12,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $6,000 |
| 5th | $5,000 |
| 6th-10th | $1,800 each |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Multi-Instance Learning (MIL) framework
- EfficientNet and ConvNeXt backbones
- Tile-based processing with attention pooling
- Careful handling of outlier class

### 2nd Place Solution

**Technique**:
- Vision Transformer on image tiles
- Hierarchical aggregation
- Out-of-distribution detection for "Other"

### 3rd Place Solution

**Innovation**:
- Self-supervised pretraining on histopathology
- Multiple magnification levels
- Ensemble with diversity

---

## Common Winning Strategies

### 1. Tile Extraction from WSIs

```python
import openslide
import numpy as np

def extract_tiles(wsi_path, tile_size=512, stride=256, level=0):
    """Extract tiles from whole slide image"""
    slide = openslide.OpenSlide(wsi_path)

    width, height = slide.level_dimensions[level]
    tiles = []
    coords = []

    for y in range(0, height - tile_size, stride):
        for x in range(0, width - tile_size, stride):
            tile = slide.read_region((x, y), level, (tile_size, tile_size))
            tile = np.array(tile.convert('RGB'))

            # Filter out background (mostly white)
            if not is_background(tile):
                tiles.append(tile)
                coords.append((x, y))

    return tiles, coords

def is_background(tile, threshold=0.8):
    """Check if tile is mostly background"""
    gray = np.mean(tile, axis=2)
    white_ratio = np.mean(gray > 220) / gray.size
    return white_ratio > threshold
```

### 2. Multi-Instance Learning Architecture

```python
import torch
import torch.nn as nn
import timm

class MILModel(nn.Module):
    def __init__(self, num_classes=6, backbone='efficientnet_b0'):
        super().__init__()

        # Feature extractor for tiles
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features

        # Attention mechanism for pooling
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, tiles):
        # tiles: (batch, num_tiles, C, H, W)
        batch_size, num_tiles = tiles.shape[:2]

        # Flatten for backbone
        tiles_flat = tiles.view(-1, *tiles.shape[2:])
        features = self.backbone(tiles_flat)  # (batch * num_tiles, feature_dim)

        # Reshape
        features = features.view(batch_size, num_tiles, -1)

        # Attention pooling
        attention_weights = self.attention(features)  # (batch, num_tiles, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        aggregated = torch.sum(features * attention_weights, dim=1)

        # Classification
        logits = self.classifier(aggregated)

        return logits, attention_weights
```

### 3. Handling Outliers (Other Class)

```python
def detect_outliers(model, features, threshold=0.7):
    """
    Detect outliers using prediction confidence
    Low confidence = likely outlier
    """
    with torch.no_grad():
        logits = model.classifier(features)
        probs = torch.softmax(logits, dim=1)

    # Max probability across known classes
    max_prob, pred_class = probs[:, :-1].max(dim=1)  # Exclude "Other" during training

    # Low confidence predictions might be outliers
    is_outlier = max_prob < threshold

    # Set outlier predictions to "Other" class
    final_pred = pred_class.clone()
    final_pred[is_outlier] = 5  # Other class index

    return final_pred

# Alternative: train separate binary outlier detector
class OutlierDetector(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.network(features)
```

### 4. Data Augmentation for Histopathology

```python
import albumentations as A

def get_train_transforms():
    """Augmentations for histopathology images"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),

        # Color augmentations (important for staining variation)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        ], p=0.5),

        # Stain normalization augmentation
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),

        # Morphological augmentations
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.2),
        A.GridDistortion(p=0.2),

        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

### 5. Multi-Resolution Processing

```python
def multi_resolution_features(slide, model, levels=[0, 1, 2]):
    """Extract features at multiple magnification levels"""
    all_features = []

    for level in levels:
        tiles, _ = extract_tiles(slide, level=level)

        if len(tiles) > 0:
            tiles_tensor = torch.stack([transform(t) for t in tiles])

            with torch.no_grad():
                features = model.backbone(tiles_tensor)

            # Pool features for this level
            level_features = features.mean(dim=0)
            all_features.append(level_features)

    # Concatenate multi-resolution features
    combined = torch.cat(all_features, dim=0)
    return combined
```

---

## Technical Insights

### Ovarian Cancer Subtype Characteristics

| Subtype | Prevalence | Visual Features |
|---------|------------|-----------------|
| HGSC | ~70% | Solid growth, pleomorphic nuclei |
| CC | ~10% | Clear cytoplasm, hobnail cells |
| EC | ~10% | Glandular architecture |
| LGSC | ~5% | Papillary, low mitotic activity |
| MC | ~3% | Goblet cells, mucin |

### WSI Processing Challenges

| Challenge | Solution |
|-----------|----------|
| Massive size | Tile-based processing |
| Background tissue | Tissue detection filtering |
| Staining variation | Color normalization |
| Multiple regions | Attention pooling |
| Memory limits | Gradient checkpointing |

### Domain Shift Considerations

Test images from different hospitals than training - key for generalization:
- Different scanning equipment
- Different staining protocols
- Different tissue preparation
- Different patient populations

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | 9 hours |
| GPU Runtime | 9 hours |
| Internet | Disabled |
| Test Size | ~550 GB |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 2nd | [Discussion](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/465410) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/465527) |
| 4th | [Discussion](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/465811) |
| 5th | [Discussion](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/466017) |

---

## Citation

```bibtex
@misc{ubc-ocean,
    author = {University of British Columbia},
    title = {UBC Ovarian Cancer Subtype Classification and Outlier Detection},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/UBC-OCEAN}},
    note = {Kaggle}
}
```
