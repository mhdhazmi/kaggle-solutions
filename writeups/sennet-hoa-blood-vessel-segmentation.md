# SenNet + HOA - Hacking the Human Vasculature

> Blood Vessel Segmentation in 3D HiP-CT Scans

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Instance Segmentation / Semantic Segmentation |
| **Data Domain** | Medical Imaging / 3D Microscopy |
| **ML Approach** | Deep Learning (U-Net variants, 3D CNNs) |
| **Key Techniques** | 3D Segmentation, Patch-Based Training, Post-Processing |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $80,000 |
| **Teams** | 1,149 |
| **Timeline** | November 2023 - January 2024 |
| **Evaluation Metric** | Surface Dice |
| **Host** | Sennet + HOA |

## Problem Description

Segment blood vessels in 3D Hierarchical Phase-Contrast Tomography (HiP-CT) images of human kidneys to help build a comprehensive map of the human vasculature.

### The Challenge

- Segment blood vessels in high-resolution 3D scans
- Handle massive image volumes (thousands of slices)
- Distinguish vessels from surrounding tissue
- Work with limited training data

### Why It Matters

- **Medical Research**: Understanding vascular structure
- **Disease Study**: Vascular changes in disease states
- **Drug Development**: Delivery pathway understanding
- **Aging Research**: How vasculature changes with age

---

## Data Description

### HiP-CT Technology

Hierarchical Phase-Contrast Tomography:
- Sub-micron resolution
- Whole organ imaging
- Non-destructive 3D visualization
- Reveals vascular microstructure

### Files

| File | Description |
|------|-------------|
| `train/` | Training images and masks |
| `test/` | Test images (hidden masks) |
| `train_rles.csv` | Run-length encoded masks |

### Image Properties

| Property | Value |
|----------|-------|
| Format | TIFF slices |
| Resolution | Sub-micron |
| Dimensions | ~500 x 500 x 500+ voxels per sample |
| Tissue | Human kidney |

### Annotation

Blood vessels manually annotated by expert anatomists, stored as run-length encoded masks.

---

## Evaluation

**Metric**: Surface Dice Coefficient

```python
def surface_dice(pred_mask, gt_mask, tolerance=1):
    """
    Surface Dice measures overlap of surface boundaries
    More lenient than volumetric Dice for thin structures
    """
    # Extract surfaces (boundaries)
    pred_surface = extract_surface(pred_mask)
    gt_surface = extract_surface(gt_mask)

    # Calculate distance maps
    pred_distances = distance_transform(~gt_surface)
    gt_distances = distance_transform(~pred_surface)

    # Count surface points within tolerance
    pred_within = np.sum(gt_distances[pred_surface] <= tolerance)
    gt_within = np.sum(pred_distances[gt_surface] <= tolerance)

    # Surface Dice
    surface_dice = (pred_within + gt_within) / (
        np.sum(pred_surface) + np.sum(gt_surface)
    )

    return surface_dice
```

Higher is better. Surface Dice is more forgiving for thin tubular structures.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $20,000 |
| 2nd | $15,000 |
| 3rd | $12,000 |
| 4th | $10,000 |
| 5th | $8,000 |
| 6th-10th | $3,000 each |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- 3D U-Net with residual connections
- Multi-scale patch training
- Heavy augmentation in 3D
- Test-time augmentation ensemble

### 2nd Place Solution

**Technique**:
- nnU-Net framework
- 2.5D approach (thick slices with context)
- Connected component post-processing

### 3rd Place Solution

**Innovation**:
- Transformer-based 3D segmentation
- Vessel-specific loss functions
- Morphological post-processing

---

## Common Winning Strategies

### 1. 3D U-Net Architecture

```python
import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))
```

### 2. Patch-Based Training

```python
import numpy as np

class PatchExtractor:
    def __init__(self, patch_size=(64, 64, 64), stride=(32, 32, 32)):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, volume, mask=None):
        """Extract overlapping patches from 3D volume"""
        patches = []
        locations = []

        d, h, w = volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        for z in range(0, d - pd + 1, sd):
            for y in range(0, h - ph + 1, sh):
                for x in range(0, w - pw + 1, sw):
                    patch = volume[z:z+pd, y:y+ph, x:x+pw]
                    patches.append(patch)
                    locations.append((z, y, x))

                    if mask is not None:
                        mask_patch = mask[z:z+pd, y:y+ph, x:x+pw]
                        # Optionally filter patches with vessels
                        if mask_patch.sum() > 100:  # Min vessel content
                            yield patch, mask_patch, (z, y, x)

    def reconstruct_from_patches(self, patches, locations, volume_shape):
        """Reconstruct volume from overlapping patches"""
        volume = np.zeros(volume_shape, dtype=np.float32)
        count = np.zeros(volume_shape, dtype=np.float32)

        pd, ph, pw = self.patch_size

        for patch, (z, y, x) in zip(patches, locations):
            volume[z:z+pd, y:y+ph, x:x+pw] += patch
            count[z:z+pd, y:y+ph, x:x+pw] += 1

        # Average overlapping regions
        volume = volume / (count + 1e-8)
        return volume
```

### 3. 3D Data Augmentation

```python
import albumentations as A
from scipy.ndimage import rotate, zoom

def augment_3d_volume(volume, mask, p=0.5):
    """Apply 3D augmentations"""
    augmented_vol = volume.copy()
    augmented_mask = mask.copy()

    # Random rotation around random axis
    if np.random.random() < p:
        angle = np.random.uniform(-15, 15)
        axes = np.random.choice([(0,1), (0,2), (1,2)])
        augmented_vol = rotate(augmented_vol, angle, axes=axes, reshape=False, mode='nearest')
        augmented_mask = rotate(augmented_mask, angle, axes=axes, reshape=False, mode='nearest')

    # Random flip
    if np.random.random() < p:
        axis = np.random.choice([0, 1, 2])
        augmented_vol = np.flip(augmented_vol, axis=axis)
        augmented_mask = np.flip(augmented_mask, axis=axis)

    # Random intensity augmentation
    if np.random.random() < p:
        # Brightness
        augmented_vol = augmented_vol * np.random.uniform(0.9, 1.1)
        # Contrast
        mean = augmented_vol.mean()
        augmented_vol = (augmented_vol - mean) * np.random.uniform(0.9, 1.1) + mean

    # Add noise
    if np.random.random() < p:
        noise = np.random.normal(0, 0.02, augmented_vol.shape)
        augmented_vol = augmented_vol + noise

    return augmented_vol, augmented_mask
```

### 4. Vessel-Specific Loss Functions

```python
import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for segmentation"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target, smooth=1e-5):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        return 1 - dice

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    """Tversky loss - better for imbalanced segmentation"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives

    def forward(self, pred, target, smooth=1e-5):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()

        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return 1 - tversky
```

### 5. Post-Processing

```python
from scipy import ndimage
import numpy as np

def post_process_vessels(pred_mask, min_size=100, fill_holes=True):
    """Clean up vessel predictions"""

    # Binarize
    binary = (pred_mask > 0.5).astype(np.uint8)

    # Remove small components
    labeled, num_features = ndimage.label(binary)
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

    for i, size in enumerate(component_sizes):
        if size < min_size:
            binary[labeled == (i + 1)] = 0

    # Fill small holes
    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)

    # Optional: morphological closing to connect nearby vessels
    struct = ndimage.generate_binary_structure(3, 2)
    binary = ndimage.binary_closing(binary, structure=struct, iterations=1)

    return binary.astype(np.float32)
```

---

## Technical Insights

### 3D vs 2.5D Approaches

| Approach | Pros | Cons |
|----------|------|------|
| True 3D | Full spatial context | Memory intensive |
| 2.5D (thick slices) | Efficient | Limited depth context |
| 2D slice-by-slice | Very efficient | No inter-slice context |

### Memory Management

| Technique | Memory Saving |
|-----------|---------------|
| Patch-based training | High |
| Mixed precision (FP16) | ~50% |
| Gradient checkpointing | ~30% |
| Smaller batch size | Variable |

### Vessel Characteristics

| Property | Value |
|----------|-------|
| Structure | Tubular, branching |
| Size range | Sub-micron to mm |
| Contrast | Phase contrast enhanced |
| Challenge | Thin, tortuous structures |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | 9 hours |
| GPU Runtime | 9 hours |
| Internet | Disabled |
| GPU Memory | T4 (16GB) or P100 |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475523) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475447) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475529) |
| 4th | [Discussion](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475477) |

---

## Citation

```bibtex
@misc{sennet-hoa-blood-vessel-segmentation,
    author = {SenNet + HOA},
    title = {SenNet + HOA - Hacking the Human Vasculature},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/blood-vessel-segmentation}},
    note = {Kaggle}
}
```
