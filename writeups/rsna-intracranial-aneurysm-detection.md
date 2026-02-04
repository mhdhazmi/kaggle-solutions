# RSNA Intracranial Aneurysm Detection

> Detect the presence and location of intracranial aneurysms in multimodal imaging data

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-label Classification / Medical Image Analysis |
| **Data Domain** | Medical Imaging / Radiology (CT/MRI) |
| **ML Approach** | 3D Deep Learning (nnU-Net + Classification) |
| **Key Techniques** | Vessel Segmentation, Coarse-to-Fine Pipeline, ROI Extraction |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,147 |
| **Timeline** | July 28, 2025 - October 15, 2025 |
| **Evaluation Metric** | Mean Weighted Columnwise AUCROC |
| **Host** | Radiological Society of North America (RSNA) |

## Problem Description

Brain aneurysms (intracranial aneurysms) are focal dilations in the arteries of the brain that may not show symptoms initially but can be deadly if not diagnosed accurately and treated appropriately.

### The Challenge

- **~3% of global population** affected by intracranial aneurysms
- **~50% diagnosed only after rupture** (often fatal)
- **~500,000 deaths annually** worldwide
- **Half of victims younger than 50**

### The Task

Detect the **presence** and **location** of intracranial aneurysms in brain imaging across multiple modalities (CTA, MRA, MRI).

### Target Labels (14 total)

**13 Anatomical Locations**:
1. Left Infraclinoid Internal Carotid Artery
2. Right Infraclinoid Internal Carotid Artery
3. Left Supraclinoid Internal Carotid Artery
4. Right Supraclinoid Internal Carotid Artery
5. Left Middle Cerebral Artery
6. Right Middle Cerebral Artery
7. Anterior Communicating Artery
8. Left Anterior Cerebral Artery
9. Right Anterior Cerebral Artery
10. Left Posterior Communicating Artery
11. Right Posterior Communicating Artery
12. Basilar Tip
13. Other Posterior Circulation

**Plus**: `Aneurysm Present` (main target, weighted 13× in evaluation)

---

## Data Description

### Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Total Files** | 1,002,760 |
| **Total Size** | 311.25 GB |
| **File Types** | DICOM (.dcm), NIfTI (.nii), Python (.py) |
| **Test Set** | ~2,500 series |
| **Modalities** | CTA, MRA, T1 post-contrast MRI, T2-weighted MRI |

### Data Files

| File | Description |
|------|-------------|
| `train.csv` | Primary training labels (SeriesInstanceUID, Modality, demographics, 14 target columns) |
| `train_localizers.csv` | Aneurysm localization data (coordinates, SOPInstanceUID) |
| `series/` | DICOM series (one folder per series) |
| `segmentations/` | NIfTI vessel segmentations (subset of training data) |
| `kaggle_evaluation/` | Evaluation API files |

### Vessel Segmentation Labels

| Label Value | Anatomical Location |
|-------------|---------------------|
| 1 | Other Posterior Circulation |
| 2 | Basilar Tip |
| 3 | Right Posterior Communicating Artery |
| 4 | Left Posterior Communicating Artery |
| 5 | Right Infraclinoid Internal Carotid Artery |
| 6 | Left Infraclinoid Internal Carotid Artery |
| 7 | Right Supraclinoid Internal Carotid Artery |
| 8 | Left Supraclinoid Internal Carotid Artery |
| 9 | Right Middle Cerebral Artery |
| 10 | Left Middle Cerebral Artery |
| 11 | Right Anterior Cerebral Artery |
| 12 | Left Anterior Cerebral Artery |
| 13 | Anterior Communicating Artery |

---

## Evaluation

**Metric**: Mean Weighted Columnwise AUCROC

### Scoring Formula

```
Final Score = 0.5 × (AUC_AneurysmPresent + (1/13) × Σ AUC_location_i)
```

- **Aneurysm Present**: Weight = 13
- **Each location**: Weight = 1
- Final score = average of Aneurysm Present score and average of 13 location scores

### Submission

Online API-based evaluation - test series served one at a time in random order.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $12,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $5,000 |
| 5th-9th | $3,000 each |

**Winners invited to RSNA Annual Meeting** AI Challenge Recognition Event.

---

## Top Solutions

### 1st Place: Location-Aware Aneurysm Detection via Vessel-ROI Masking

**Key Innovation**: Coarse-to-fine pipeline using vessel segmentation to guide ROI-based classification.

#### Pipeline Overview

```
DICOM → Preprocessing → Coarse Vessel Localization → Fine Segmentation → ROI Classification
```

**1. Preprocessing**:
- Filter slices with consistent dimensions and spacing
- Convert DICOM to NIfTI using dcm2niix (with gdcmconv fallback)
- Standardize orientation using nnU-Net's SimpleITKIOWithReorient
- Z-score intensity normalization

**2. nnU-Net Segmentation (Coarse-to-Fine)**:

| Model | Spacing | Classes | Purpose |
|-------|---------|---------|---------|
| Model 1 (Coarse) | 1.0×1.0×1.0 mm | 3 vessel groups | Fast ROI localization |
| Model 2 (Fine/Balanced) | 0.80×0.45×0.44 mm | 13 locations | Precise segmentation |
| Model 3 (Fine/Recall) | 0.80×0.45×0.44 mm | 13 locations | High-sensitivity detection |

**Loss Functions**:
- Model 1: Dice + Cross-Entropy
- Model 2: Dice + CE + SkeletonRecall (weight=1)
- Model 3: Tversky + CE + SkeletonRecall (weight=3)

**Key Augmentation**: Disabled left-right mirroring (preserve anatomical asymmetry), stronger intensity/geometric augmentations, low-resolution simulation.

**Two-Stage Inference**:
1. **Stage 1 (Coarse)**: Sliding window (overlap=0.2), DBSCAN clustering to remove false positives, crop 140×140×140 mm ROI
2. **Stage 2 (Fine)**: Higher overlap (0.3), compute tight bounding box from segmentation

**3. ROI Classification**:
- Input: 128×256×256 voxel volumes
- Backbone: nnU-Net pre-trained on vessel segmentation
- Simplified decoder (removed final block for efficiency)
- Auxiliary detection task for improved training

#### Key Design Principles

1. **Coarse-to-Fine Efficiency**: Fast low-res model finds candidate region, detailed models focus on ROI
2. **Segmentation as Structural Guide**: Vessel masks help classifier focus on relevant anatomy
3. **Data Quality Control**: Excluded ~60 series with quality issues

#### Data Preparation
- Multilabel-stratified 5-fold cross-validation
- Excluded series with orientation anomalies, corrupted DICOMs, implausible slice spacing

---

## Common Winning Strategies

### 1. Coarse-to-Fine Pipeline
Fast initial scan to localize ROI, then detailed analysis within the region.

### 2. Vessel Segmentation as Prior
Using explicit vessel masks guides the classifier to focus on anatomically relevant regions.

### 3. nnU-Net Framework
Pre-trained nnU-Net models provide strong 3D medical image segmentation foundation.

### 4. Multi-Model Ensemble
Combining balanced and recall-focused models improves overall detection.

### 5. SkeletonRecall Loss
Improves connectivity of thin vessel structures that standard losses miss.

### 6. Anatomical Asymmetry Preservation
Disabling left-right augmentation preserves important anatomical laterality information.

---

## Technical Insights

### Why Location-Aware Detection Works

1. **Anatomical specificity**: Aneurysms occur at specific vessel locations
2. **Segmentation guidance**: Vessel masks provide structural context
3. **ROI focus**: Reduces false positives from non-vascular structures

### Data Quality Considerations

Common issues requiring exclusion:
- Orientation anomalies
- Corrupted DICOM files
- Implausible slice spacing
- Inconsistent image dimensions

### Multi-Modality Handling

Different imaging modalities (CTA, MRA, MRI) require:
- Modality-specific preprocessing
- Intensity normalization strategies
- Potentially separate model branches

---

## Technical Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | ≤12 hours |
| GPU Runtime | ≤12 hours |
| Internet Access | Disabled |
| External Data | Allowed (including pre-trained models) |

---

## Solution Links

| Place | Author | Solution |
|-------|--------|----------|
| 1st | - | [Writeup](https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution) |
| 2nd | - | [Writeup](https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) |
| 3rd | - | [Writeup](https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution) |

---

## Acknowledgements

**Contributing Institutions**:
- Aga Khan University Hospital (Pakistan)
- Duke University (USA)
- University of California San Francisco (USA)
- Stanford University (USA)
- And many more international institutions

**Professional Societies**: ASNR, SNIS, ESNR

---

## Citation

```bibtex
@misc{rsna-intracranial-aneurysm-detection,
    author = {Jeff Rudie and Evan Calabrese and Robyn Ball and Peter Chang and others},
    title = {RSNA Intracranial Aneurysm Detection},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/rsna-intracranial-aneurysm-detection}},
    note = {Kaggle, Radiological Society of North America}
}
```
