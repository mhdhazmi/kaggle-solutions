# BirdCLEF 2024

> Bird species identification from audio, focused on under-studied species in the Western Ghats, India

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Label Audio Classification |
| **Data Domain** | Bioacoustics / Wildlife Conservation |
| **ML Approach** | CNN on Spectrograms + Transformers |
| **Key Techniques** | Mel Spectrograms, Data Augmentation, Semi-Supervised Learning |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 974 |
| **Timeline** | 2024 |
| **Evaluation Metric** | BirdCLEF ROC AUC |
| **Host** | Cornell Lab of Ornithology |

## Problem Description

Identify under-studied Indian bird species from audio recordings, enabling automated biodiversity monitoring in the Western Ghats biodiversity hotspot.

### The Challenge

- Identify bird species from continuous audio
- Handle limited training data for rare species
- Process soundscape recordings with multiple overlapping species
- Generalize to field conditions (noise, distance)

### Why It Matters

- **Biodiversity**: Western Ghats is a major biodiversity hotspot
- **Conservation**: Monitor habitat restoration success
- **Scalability**: PAM can sample larger areas than observers
- **Climate**: Track biodiversity changes over time

---

## Data Description

### Training Data

| Component | Description |
|-----------|-------------|
| Audio recordings | Bird calls and songs |
| Species labels | Target bird species |
| Metadata | Location, recordist, quality |

### Target Species

Focus on under-studied species of the Western Ghats, India - one of the world's most biodiverse regions.

### Audio Characteristics

| Property | Value |
|----------|-------|
| Sample rate | 32kHz typical |
| Format | OGG/WAV |
| Duration | Variable (5 seconds to hours) |
| Challenge | Background noise, overlapping calls |

---

## Evaluation

**Metric**: BirdCLEF ROC AUC

Area under the ROC curve averaged across species.

```python
from sklearn.metrics import roc_auc_score

# Per-species AUC then averaged
species_aucs = []
for species in species_list:
    y_true_sp = (y_true == species).astype(int)
    y_pred_sp = predictions[:, species_idx]
    auc = roc_auc_score(y_true_sp, y_pred_sp)
    species_aucs.append(auc)

final_auc = np.mean(species_aucs)
```

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Author**: Kirill Chemrov

**Key Approach**:
- EfficientNet on mel spectrograms
- Heavy data augmentation
- Pseudo-labeling
- Ensemble of models

### 2nd Place Solution

**Technique**:
- Transformer-based audio models
- Multi-scale features
- Knowledge distillation

### 5th Place Solution

**Innovation**:
- BirdNET integration
- Transfer learning
- Species-specific thresholds

---

## Common Winning Strategies

### 1. Mel Spectrogram Generation

```python
import librosa
import numpy as np

def audio_to_melspec(audio_path, sr=32000, n_mels=128, fmax=16000):
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)

    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        hop_length=512,
        n_fft=2048
    )

    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db
```

### 2. CNN Architecture

```python
import torch.nn as nn
import timm

class BirdClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1,  # Grayscale spectrogram
            num_classes=num_classes
        )

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))
```

### 3. Audio Data Augmentation

```python
import audiomentations as A

augment = A.Compose([
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    A.Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

# Also spectrogram augmentations
spec_augment = nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)
```

### 4. Handling Soundscapes

```python
def process_soundscape(audio_path, model, window_size=5, hop_size=2.5):
    """Process continuous audio in windows"""
    y, sr = librosa.load(audio_path, sr=32000)
    duration = len(y) / sr

    predictions = []
    for start in np.arange(0, duration - window_size, hop_size):
        end = start + window_size
        window = y[int(start * sr):int(end * sr)]
        mel_spec = audio_to_melspec(window)
        pred = model(mel_spec.unsqueeze(0))
        predictions.append({
            'start': start,
            'end': end,
            'predictions': pred
        })

    return predictions
```

### 5. Pseudo-Labeling

```python
def pseudo_label(model, unlabeled_data, threshold=0.9):
    """Generate pseudo-labels for confident predictions"""
    pseudo_labels = []

    model.eval()
    with torch.no_grad():
        for audio in unlabeled_data:
            pred = model(audio)

            # Only use high-confidence predictions
            max_prob = pred.max()
            if max_prob > threshold:
                pseudo_labels.append({
                    'audio': audio,
                    'label': pred.argmax(),
                    'confidence': max_prob
                })

    return pseudo_labels
```

---

## Technical Insights

### Spectrogram Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| n_mels | 128-256 | Frequency resolution |
| fmax | 16000 | Maximum frequency |
| n_fft | 2048 | Time-frequency tradeoff |
| hop_length | 512 | Time resolution |

### Challenges in Bird Audio

| Challenge | Solution |
|-----------|----------|
| Background noise | Noise augmentation, filtering |
| Overlapping calls | Multi-label classification |
| Rare species | Oversampling, pseudo-labeling |
| Domain shift | Train-test augmentation alignment |

### Transfer Learning Sources

| Source | Use |
|--------|-----|
| ImageNet | CNN pre-training |
| BirdNET | Bird-specific embeddings |
| AudioSet | Audio event classification |
| Previous BirdCLEF | Similar task |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | 2 hours |
| Internet | Disabled |
| GPU | Available |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/birdclef-2024/discussion/512197) |
| 2nd | [Discussion](https://www.kaggle.com/c/birdclef-2024/discussion/512340) |
| 3rd | [Discussion](https://www.kaggle.com/c/birdclef-2024/discussion/511905) |
| 4th | [Discussion](https://www.kaggle.com/c/birdclef-2024/discussion/511845) |
| 5th | [Discussion](https://www.kaggle.com/c/birdclef-2024/discussion/511535) |

---

## Citation

```bibtex
@misc{birdclef-2024,
    author = {Cornell Lab of Ornithology},
    title = {BirdCLEF 2024},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/birdclef-2024}},
    note = {Kaggle}
}
```
