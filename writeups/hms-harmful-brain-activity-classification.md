# HMS - Harmful Brain Activity Classification

> Classify seizures and other patterns of harmful brain activity in critically ill patients

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Class Classification |
| **Data Domain** | Medical / Neuroscience / EEG Analysis |
| **ML Approach** | Deep Learning on Spectrograms + EEG Signals |
| **Key Techniques** | Signal Processing, CNNs, Transformers, Multi-Input Models |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 2,767 |
| **Timeline** | January - April 2024 |
| **Evaluation Metric** | KL Divergence |
| **Host** | Harvard Medical School |

## Problem Description

Detect and classify seizures and other harmful brain activity patterns from EEG recordings of critically ill hospital patients.

### The Challenge

- Classify 6 types of brain activity from EEG data
- Handle expert disagreement in labels
- Work with both raw EEG and spectrograms
- Predict probability distributions (soft labels)

### The 6 Brain Activity Classes

| Class | Abbreviation | Description |
|-------|--------------|-------------|
| Seizure | SZ | Epileptic seizure activity |
| Lateralized Periodic Discharges | LPD | One-sided periodic patterns |
| Generalized Periodic Discharges | GPD | Bilateral periodic patterns |
| Lateralized Rhythmic Delta Activity | LRDA | One-sided slow waves |
| Generalized Rhythmic Delta Activity | GRDA | Bilateral slow waves |
| Other | Other | No specific pattern |

### Why It Matters

- **Patient Care**: Faster detection enables quicker treatment
- **Scalability**: Reduce bottleneck of manual EEG review
- **Accuracy**: Reduce fatigue-related errors and inter-reviewer variability
- **Research**: Enable drug development for seizure prevention

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Metadata with labels and annotation info |
| `train_eegs/` | Raw EEG data (parquet files) |
| `train_spectrograms/` | Pre-computed spectrograms (parquet) |
| `test.csv` | Test metadata |

### Label Structure

Expert annotators reviewed 50-second EEG samples with matched 10-minute spectrograms. Labels represent vote distributions:

| Column | Description |
|--------|-------------|
| `seizure_vote` | Number of experts labeling as seizure |
| `lpd_vote` | Votes for LPD |
| `gpd_vote` | Votes for GPD |
| `lrda_vote` | Votes for LRDA |
| `grda_vote` | Votes for GRDA |
| `other_vote` | Votes for Other |
| `expert_consensus` | Mode of expert votes |

### EEG Channels

Standard 10-20 system with channels like:
- Fp1, Fp2 (frontal)
- F3, F4, F7, F8 (frontal)
- T3, T4, T5, T6 (temporal)
- C3, C4, Cz (central)
- P3, P4 (parietal)
- O1, O2 (occipital)

---

## Evaluation

**Metric**: KL Divergence

```python
import numpy as np

def kl_divergence(y_true, y_pred, epsilon=1e-15):
    """
    y_true: Ground truth distributions (N x 6)
    y_pred: Predicted distributions (N x 6)
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Normalize to ensure valid probability distributions
    y_true = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    # KL divergence
    kl = np.sum(y_true * np.log(y_true / y_pred), axis=1)

    return np.mean(kl)
```

Lower is better. Predicting soft probability distributions matching expert vote proportions.

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
- EfficientNet on spectrograms
- 1D CNN on raw EEG
- Multi-input fusion architecture
- Heavy augmentation on both modalities

### 2nd Place Solution

**Technique**:
- Dual-branch model (EEG + spectrogram)
- WaveNet-style dilated convolutions for EEG
- Attention mechanisms for channel selection

### 3rd Place Solution

**Innovation**:
- Transformer encoder for EEG sequences
- Spectrogram patches with ViT
- Label smoothing for soft targets

---

## Common Winning Strategies

### 1. EEG Preprocessing

```python
import numpy as np
from scipy import signal

def preprocess_eeg(eeg_data, sample_rate=200):
    """Standard EEG preprocessing"""
    processed = {}

    for channel in eeg_data.columns:
        data = eeg_data[channel].values

        # Remove DC offset
        data = data - np.mean(data)

        # Bandpass filter (0.5-40 Hz typical for EEG)
        nyq = sample_rate / 2
        low = 0.5 / nyq
        high = 40 / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        data = signal.filtfilt(b, a, data)

        # Notch filter for powerline noise (50/60 Hz)
        b_notch, a_notch = signal.iirnotch(50, 30, sample_rate)
        data = signal.filtfilt(b_notch, a_notch, data)

        # Clip extreme values
        data = np.clip(data, -1000, 1000)

        processed[channel] = data

    return processed
```

### 2. Spectrogram Generation

```python
import librosa
import numpy as np

def create_spectrogram(eeg_channel, sample_rate=200, n_mels=128):
    """Create mel spectrogram from EEG channel"""
    # Short-time Fourier transform
    spec = librosa.feature.melspectrogram(
        y=eeg_channel,
        sr=sample_rate,
        n_mels=n_mels,
        fmax=40,  # Max frequency for EEG
        hop_length=sample_rate // 4
    )

    # Convert to log scale
    spec_db = librosa.power_to_db(spec, ref=np.max)

    return spec_db

def create_multi_channel_spectrogram(eeg_data, channels):
    """Stack spectrograms from multiple channels"""
    specs = []
    for channel in channels:
        spec = create_spectrogram(eeg_data[channel])
        specs.append(spec)

    return np.stack(specs, axis=0)
```

### 3. Dual-Input Model Architecture

```python
import torch
import torch.nn as nn
import timm

class DualInputModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Spectrogram branch (2D CNN)
        self.spec_backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            in_chans=4,  # 4 spectrogram channels
            num_classes=0  # Remove classifier
        )
        spec_features = self.spec_backbone.num_features

        # EEG branch (1D CNN)
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        eeg_features = 128

        # Fusion and classifier
        self.classifier = nn.Sequential(
            nn.Linear(spec_features + eeg_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, spectrogram, eeg):
        spec_feat = self.spec_backbone(spectrogram)
        eeg_feat = self.eeg_encoder(eeg).squeeze(-1)

        combined = torch.cat([spec_feat, eeg_feat], dim=1)
        return self.classifier(combined)
```

### 4. Data Augmentation

```python
import albumentations as A
from audiomentations import Compose, AddGaussianNoise, TimeStretch

# Spectrogram augmentations
spec_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
])

# EEG signal augmentations
def augment_eeg(eeg_data, p=0.5):
    if np.random.random() < p:
        # Random amplitude scaling
        scale = np.random.uniform(0.8, 1.2)
        eeg_data = eeg_data * scale

    if np.random.random() < p:
        # Add Gaussian noise
        noise = np.random.normal(0, 0.1, eeg_data.shape)
        eeg_data = eeg_data + noise

    if np.random.random() < p:
        # Random channel dropout
        drop_channels = np.random.choice(eeg_data.shape[0], size=2, replace=False)
        eeg_data[drop_channels] = 0

    return eeg_data
```

### 5. Handling Soft Labels

```python
def soft_label_loss(predictions, vote_counts):
    """
    KL divergence loss for soft labels
    vote_counts: [seizure, lpd, gpd, lrda, grda, other]
    """
    # Convert votes to probability distribution
    total_votes = vote_counts.sum(dim=1, keepdim=True)
    target_probs = vote_counts / total_votes

    # KL divergence
    log_preds = torch.log(predictions + 1e-8)
    kl_div = torch.sum(target_probs * (torch.log(target_probs + 1e-8) - log_preds), dim=1)

    return kl_div.mean()

# Alternative: weighted cross-entropy
def weighted_ce_loss(predictions, vote_counts):
    """Weighted cross-entropy treating each expert vote as a sample"""
    total_votes = vote_counts.sum(dim=1, keepdim=True)
    weights = vote_counts / total_votes

    log_preds = torch.log(predictions + 1e-8)
    loss = -torch.sum(weights * log_preds, dim=1)

    return loss.mean()
```

---

## Technical Insights

### EEG Pattern Characteristics

| Pattern | Frequency | Spatial | Duration |
|---------|-----------|---------|----------|
| Seizure | Variable, rhythmic | Can spread | Seconds-minutes |
| LPD | 0.5-3 Hz periodic | Lateralized | Persistent |
| GPD | 0.5-3 Hz periodic | Bilateral | Persistent |
| LRDA | 0.5-4 Hz rhythmic | Lateralized | Persistent |
| GRDA | 0.5-4 Hz rhythmic | Bilateral | Persistent |

### Expert Agreement Levels

| Type | Description |
|------|-------------|
| Idealized | High agreement among experts |
| Proto | ~50% "Other", ~50% specific pattern |
| Edge cases | Split between 2 named patterns |

### Model Input Considerations

| Input Type | Pros | Cons |
|------------|------|------|
| Raw EEG | Full information | High dimensional |
| Spectrogram | Visual patterns | Loses phase info |
| Both | Best of both | Complex model |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | 9 hours |
| GPU Runtime | 9 hours |
| Internet | Disabled |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492232) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492304) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492155) |
| 4th | [Discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492367) |
| 5th | [Discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492178) |

---

## Citation

```bibtex
@misc{hms-harmful-brain-activity-classification,
    author = {Harvard Medical School},
    title = {HMS - Harmful Brain Activity Classification},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/hms-harmful-brain-activity-classification}},
    note = {Kaggle}
}
```
