# BirdCLEF+ 2025

> Species identification from audio, focused on birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia.

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-label Audio Classification |
| **Data Domain** | Audio / Bioacoustics / Wildlife |
| **ML Approach** | CNN + Transformers on Spectrograms |
| **Key Techniques** | Mel Spectrograms, Data Augmentation, Semi-supervised Learning |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 2,025 |
| **Timeline** | March - June 2025 |
| **Evaluation Metric** | ROC AUC |
| **Host** | Cornell Lab of Ornithology |

## Problem Description

Identify species from different taxonomic groups (birds, amphibians, mammals, insects) by their acoustic signatures in soundscape recordings from El Silencio Natural Reserve, Colombia.

### The Challenge

- Process continuous audio data (1-minute soundscapes)
- Recognize multiple species simultaneously
- Train reliable classifiers with limited labeled data
- Handle rare and endangered species with few examples

### Why It Matters

- **Conservation**: Monitor biodiversity in tropical rainforests
- **Restoration**: Evaluate ecological restoration project success
- **Research**: Automate species detection at scale
- **Protection**: Support efforts to protect endangered ecosystems

---

## Data Description

### Dataset Structure

| Component | Description |
|-----------|-------------|
| `train_audio/` | Short recordings of individual species from Xeno-canto, iNaturalist, and Colombian Sound Archive |
| `test_soundscapes/` | ~700 one-minute recordings for scoring (32 kHz, ogg format) |
| `train_soundscapes/` | Unlabeled audio from same locations as test |
| `train.csv` | Metadata including species labels, coordinates, quality ratings |

### Key Features

| Field | Description |
|-------|-------------|
| `primary_label` | Species code (eBird code for birds, iNaturalist ID for others) |
| `secondary_labels` | Additional species in recording |
| `latitude/longitude` | Recording location (for dialect variations) |
| `rating` | Quality rating 1-5 (Xeno-canto only) |
| `collection` | Source: XC (Xeno-canto), iNat, or CSA |

### Taxonomic Groups

- **Birds**: Most represented group
- **Amphibians**: Including endangered species
- **Mammals**: Various vocalizing species
- **Insects**: Acoustic patterns

### Technical Specifications

- Audio resampled to **32 kHz**
- Format: **OGG**
- Test soundscapes: **1 minute** duration each

---

## Evaluation

**Metric**: ROC AUC (Area Under ROC Curve)

Multi-label classification evaluated per species, then averaged.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |

**Total Prize Pool**: $50,000

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Multi-model ensemble with spectrogram-based CNNs
- Efficient use of unlabeled soundscape data
- Strong augmentation pipeline

### 2nd Place Solution

**Approach**:
- Transformer-based architecture
- Focal loss for class imbalance
- Pseudo-labeling on unlabeled data

### 3rd Place Solution

**Technique**:
- EfficientNet backbone
- Mixup and SpecAugment
- Careful threshold tuning

---

## Common Winning Strategies

### 1. Spectrogram Conversion

Converting audio to mel spectrograms:
```
Audio → Mel Spectrogram → CNN/Transformer → Predictions
```

Key parameters:
- Sample rate: 32,000 Hz
- n_mels: 128-256
- hop_length: 512
- fmax: 16,000 Hz

### 2. Backbone Architectures

| Architecture | Strengths |
|--------------|-----------|
| EfficientNet | Efficient, good accuracy |
| ConvNeXt | Modern CNN design |
| BEATs | Audio-specific transformer |
| AST | Audio Spectrogram Transformer |

### 3. Data Augmentation

- **Mixup**: Blend multiple audio samples
- **SpecAugment**: Mask time/frequency bands
- **Time shifting**: Random temporal shifts
- **Noise injection**: Add background noise

### 4. Semi-supervised Learning

Leveraging unlabeled soundscapes:
- Pseudo-labeling with confident predictions
- Self-training iterations
- Consistency regularization

### 5. Class Imbalance Handling

- Focal loss for rare species
- Oversampling minority classes
- Class-balanced sampling

---

## Technical Insights

### Audio Processing Pipeline

```
1. Load OGG file (librosa/soundfile)
2. Resample to 32 kHz if needed
3. Convert to mel spectrogram
4. Normalize (per-channel or global)
5. Crop/pad to fixed length (5-10 sec chunks)
6. Apply augmentations
7. Feed to model
```

### Challenge: Few-Shot Learning

Many species have very few training examples:
- Data augmentation becomes critical
- Transfer learning from pretrained audio models
- Metric learning approaches

### Challenge: Multi-Label Nature

Multiple species can call simultaneously:
- Binary cross-entropy per class
- Sigmoid activation (not softmax)
- Per-class threshold optimization

---

## Code Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Notebook | ≤12 hours |
| GPU Notebook | ≤12 hours |
| Internet | Disabled during inference |
| External Data | Allowed (pre-trained models OK) |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/birdclef-2025/discussion/583577) |
| 2nd | [Discussion](https://www.kaggle.com/c/birdclef-2025/discussion/583699) |
| 3rd | [Discussion](https://www.kaggle.com/c/birdclef-2025/discussion/583477) |
| 4th | [Discussion](https://www.kaggle.com/c/birdclef-2025/discussion/584034) |
| 5th | [Discussion](https://www.kaggle.com/c/birdclef-2025/discussion/583312) |

---

## Citation

```bibtex
@misc{birdclef-2025,
    title = {BirdCLEF+ 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/birdclef-2025}},
    note = {Cornell Lab of Ornithology, Kaggle}
}
```
