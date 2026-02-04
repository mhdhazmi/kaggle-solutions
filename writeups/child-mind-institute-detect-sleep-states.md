# Child Mind Institute - Detect Sleep States

> Detect sleep onset and wake from wrist-worn accelerometer data

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Event Detection / Time Series Classification |
| **Data Domain** | Health / Sleep Research |
| **ML Approach** | Deep Learning on Time Series |
| **Key Techniques** | 1D CNNs, Transformers, Event Detection |
| **Difficulty** | Intermediate-Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,877 |
| **Timeline** | September - December 2023 |
| **Evaluation Metric** | Event Detection Average Precision |
| **Host** | Child Mind Institute |

## Problem Description

Detect when a person falls asleep (onset) and wakes up from accelerometer data - enabling large-scale sleep research without sleep labs.

### The Challenge

- Detect two event types: sleep onset and wake
- Work with multi-day continuous recordings
- Handle noisy real-world accelerometer data
- Generalize across different individuals

### Why It Matters

- **Sleep Research**: Enable large-scale studies
- **Child Health**: Understand sleep patterns in children
- **Mental Health**: Sleep impacts mood and behavior
- **Accessibility**: No need for expensive sleep labs

---

## Data Description

### Accelerometer Data

Wrist-worn devices recording at 5-second intervals:
- **ENMO**: Euclidean Norm Minus One (activity measure)
- **anglez**: Arm angle relative to vertical
- **step**: Detected step count (if available)

### Files

| File | Description |
|------|-------------|
| `train_series.parquet` | Time series accelerometer data |
| `train_events.csv` | Sleep onset and wake events |
| `test_series.parquet` | Test time series |
| `sample_submission.csv` | Submission format |

### Event Types

| Event | Description |
|-------|-------------|
| onset | Moment of falling asleep |
| wakeup | Moment of waking up |

### Data Properties

| Property | Value |
|----------|-------|
| Sampling | Every 5 seconds |
| Recording length | Multiple days per subject |
| Training subjects | ~280 |
| Events per night | 1 onset, 1 wakeup |

---

## Evaluation

**Metric**: Event Detection Average Precision

```python
def event_detection_ap(y_true_events, y_pred_events, tolerance_steps):
    """
    AP for event detection with tolerance window

    y_true_events: list of (timestamp, event_type)
    y_pred_events: list of (timestamp, score, event_type)
    tolerance_steps: number of steps for matching window
    """
    # For each event type
    for event_type in ['onset', 'wakeup']:
        true_events = [e for e in y_true_events if e[1] == event_type]
        pred_events = sorted(
            [e for e in y_pred_events if e[2] == event_type],
            key=lambda x: -x[1]  # Sort by score descending
        )

        # Match predictions to ground truth within tolerance
        matched = set()
        precisions = []
        recalls = []

        for i, (pred_time, score, _) in enumerate(pred_events):
            # Find if any true event is within tolerance
            is_match = False
            for j, (true_time, _) in enumerate(true_events):
                if j not in matched and abs(pred_time - true_time) <= tolerance_steps:
                    matched.add(j)
                    is_match = True
                    break

            # Calculate precision and recall at this point
            tp = len(matched)
            precision = tp / (i + 1)
            recall = tp / len(true_events) if true_events else 0

            precisions.append(precision)
            recalls.append(recall)

    # Calculate AP
    ap = calculate_ap(precisions, recalls)
    return ap
```

Higher is better.

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
- Transformer-based sequence model
- Multi-scale feature extraction
- Ensemble of different architectures
- Custom post-processing for event detection

### 2nd Place Solution

**Technique**:
- 1D U-Net for segmentation approach
- Treat as binary classification per timestep
- Peak detection for events

### 3rd Place Solution

**Innovation**:
- GRU with attention
- Sliding window approach
- Probabilistic event modeling

---

## Common Winning Strategies

### 1. Feature Engineering

```python
import numpy as np
import pandas as pd

def engineer_features(df):
    """Create features from accelerometer data"""

    # Rolling statistics
    windows = [12, 60, 180, 360]  # 1min, 5min, 15min, 30min at 5sec intervals

    for window in windows:
        # ENMO statistics
        df[f'enmo_mean_{window}'] = df['enmo'].rolling(window, center=True).mean()
        df[f'enmo_std_{window}'] = df['enmo'].rolling(window, center=True).std()
        df[f'enmo_max_{window}'] = df['enmo'].rolling(window, center=True).max()

        # Anglez statistics
        df[f'anglez_mean_{window}'] = df['anglez'].rolling(window, center=True).mean()
        df[f'anglez_std_{window}'] = df['anglez'].rolling(window, center=True).std()
        df[f'anglez_range_{window}'] = (
            df['anglez'].rolling(window, center=True).max() -
            df['anglez'].rolling(window, center=True).min()
        )

    # Activity indicators
    df['is_active'] = (df['enmo'] > 0.05).astype(int)
    df['activity_fraction_60'] = df['is_active'].rolling(60, center=True).mean()

    # Arm position indicators
    df['arm_horizontal'] = (np.abs(df['anglez']) < 20).astype(int)
    df['arm_position_stable'] = (df[f'anglez_std_60'] < 10).astype(int)

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Change detection
    df['enmo_diff'] = df['enmo'].diff()
    df['anglez_diff'] = df['anglez'].diff()

    return df
```

### 2. Segmentation Approach (U-Net style)

```python
import torch
import torch.nn as nn

class SleepSegmentationModel(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(input_channels, hidden_channels)
        self.enc2 = ConvBlock(hidden_channels, hidden_channels * 2)
        self.enc3 = ConvBlock(hidden_channels * 2, hidden_channels * 4)

        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(hidden_channels * 4, hidden_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose1d(hidden_channels * 8, hidden_channels * 4, 2, 2)
        self.dec3 = ConvBlock(hidden_channels * 8, hidden_channels * 4)

        self.up2 = nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, 2, 2)
        self.dec2 = ConvBlock(hidden_channels * 4, hidden_channels * 2)

        self.up1 = nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, 2, 2)
        self.dec1 = ConvBlock(hidden_channels * 2, hidden_channels)

        # Output: probability of being in sleep state
        self.output = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.output(d1))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)
```

### 3. Event Detection from Predictions

```python
from scipy.signal import find_peaks
import numpy as np

def detect_events(sleep_probs, timestamps, min_sleep_hours=3):
    """
    Detect onset and wake events from sleep probability curve

    sleep_probs: probability of being asleep at each timestep
    timestamps: corresponding timestamps
    """
    events = []

    # Smooth predictions
    sleep_probs_smooth = np.convolve(sleep_probs, np.ones(12)/12, mode='same')

    # Find transitions
    sleep_diff = np.diff(sleep_probs_smooth)

    # Onset: large positive change (falling asleep)
    onset_candidates, onset_props = find_peaks(sleep_diff, height=0.3, distance=720)

    # Wake: large negative change (waking up)
    wake_candidates, wake_props = find_peaks(-sleep_diff, height=0.3, distance=720)

    # Match onsets with wakes to form valid sleep periods
    for onset_idx in onset_candidates:
        onset_time = timestamps[onset_idx]

        # Find next wake event
        future_wakes = wake_candidates[wake_candidates > onset_idx]
        if len(future_wakes) > 0:
            wake_idx = future_wakes[0]
            wake_time = timestamps[wake_idx]

            # Check if sleep duration is reasonable
            sleep_duration = (wake_time - onset_time).total_seconds() / 3600
            if sleep_duration >= min_sleep_hours and sleep_duration <= 14:
                events.append({
                    'event': 'onset',
                    'step': onset_idx,
                    'score': onset_props['peak_heights'][list(onset_candidates).index(onset_idx)]
                })
                events.append({
                    'event': 'wakeup',
                    'step': wake_idx,
                    'score': wake_props['peak_heights'][list(wake_candidates).index(wake_idx)]
                })

    return events
```

### 4. Transformer Model

```python
class SleepTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=128, nhead=8, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Event detection heads
        self.onset_head = nn.Linear(d_model, 1)
        self.wake_head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        features = self.transformer(x, src_key_padding_mask=mask)

        onset_logits = self.onset_head(features).squeeze(-1)
        wake_logits = self.wake_head(features).squeeze(-1)

        return torch.sigmoid(onset_logits), torch.sigmoid(wake_logits)
```

### 5. Training with Event Labels

```python
def create_target_labels(events_df, series_length, tolerance=30):
    """
    Create soft labels around event times

    events_df: dataframe with event times
    series_length: length of time series
    tolerance: steps around event to label positive
    """
    onset_labels = np.zeros(series_length)
    wake_labels = np.zeros(series_length)

    for _, event in events_df.iterrows():
        step = event['step']
        event_type = event['event']

        # Create Gaussian-like soft label around event
        indices = np.arange(max(0, step - tolerance), min(series_length, step + tolerance + 1))
        distances = np.abs(indices - step)
        weights = np.exp(-distances**2 / (2 * (tolerance/3)**2))

        if event_type == 'onset':
            onset_labels[indices] = np.maximum(onset_labels[indices], weights)
        else:
            wake_labels[indices] = np.maximum(wake_labels[indices], weights)

    return onset_labels, wake_labels
```

---

## Technical Insights

### Sleep Detection Indicators

| Feature | Awake | Asleep |
|---------|-------|--------|
| ENMO | Higher, variable | Low, stable |
| anglez | Variable | Stable (lying down) |
| Movement | Frequent | Rare |

### Typical Sleep Patterns

- Adults: 7-9 hours continuous
- Children: 9-12 hours, may have naps
- Onset: Usually 9PM - 12AM
- Wake: Usually 5AM - 9AM

### Data Challenges

| Challenge | Solution |
|-----------|----------|
| Missing data | Interpolation, masking |
| Multi-day series | Split into nights |
| Naps | Allow multiple events |
| Artifacts | Robust statistics |

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
| 1st | [Discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459715) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459627) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459599) |
| 4th | [Discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459637) |
| 5th | [Discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459766) |

---

## Citation

```bibtex
@misc{child-mind-institute-detect-sleep-states,
    author = {Child Mind Institute},
    title = {Child Mind Institute - Detect Sleep States},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/child-mind-institute-detect-sleep-states}},
    note = {Kaggle}
}
```
