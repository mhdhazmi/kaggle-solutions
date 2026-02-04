# Stanford Ribonanza RNA Folding

> Create a model that predicts the structures of any RNA molecule

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Sequence-to-Sequence Regression |
| **Data Domain** | Computational Biology / RNA |
| **ML Approach** | Transformers + Convolutional Models |
| **Key Techniques** | Sequence Modeling, Chemical Reactivity Prediction |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Prediction Competition |
| **Total Prize** | $100,000 |
| **Teams** | 755 |
| **Timeline** | September - December 2023 |
| **Evaluation Metric** | Mean Absolute Error |
| **Host** | Stanford University |

## Problem Description

Predict chemical reactivity profiles for RNA molecules - measurements that reveal RNA structure without directly solving the 3D structure.

### The Challenge

- Predict reactivity at each position in RNA sequence
- Handle variable-length sequences
- Predict for two chemical modifiers (DMS, 2A3)
- Generalize to novel RNA structures

### Why It Matters

- **Medicine**: Design better mRNA vaccines
- **Drug Discovery**: RNA-based therapeutics
- **Basic Science**: Understanding life's molecular machinery
- **Climate**: Biological carbon sequestration

---

## Data Description

### Chemical Mapping Experiments

| Modifier | Full Name | What It Measures |
|----------|-----------|------------------|
| DMS | Dimethyl sulfate | Unpaired A and C bases |
| 2A3 | 2-aminopyridine-3-carboxylic acid | Unpaired positions |

### Files

| File | Description |
|------|-------------|
| `train_data.csv` | Training reactivity profiles |
| `test_sequences.csv` | Test sequences |
| `sample_submission.csv` | Submission format |

### Data Fields

| Column | Description |
|--------|-------------|
| `sequence_id` | Unique identifier |
| `sequence` | RNA sequence (A, C, G, U) |
| `experiment_type` | DMS_MaP or 2A3_MaP |
| `reactivity_0001...` | Reactivity at each position |
| `signal_to_noise` | Data quality measure |

### RNA Sequences

- Lengths: ~100-500 nucleotides
- Alphabet: A, C, G, U
- Training: ~1.5M sequences
- Target: Reactivity profile (float per position)

---

## Evaluation

**Metric**: Mean Absolute Error

```python
import numpy as np

def mae_score(y_true, y_pred):
    """
    MAE calculated on valid (non-null) positions only
    """
    mask = ~np.isnan(y_true)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))
```

Lower is better. Only scored on positions with experimental data.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $30,000 |
| 2nd | $20,000 |
| 3rd | $15,000 |
| 4th | $12,000 |
| 5th | $10,000 |
| 6th-10th | $2,600 each |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Large transformer architecture
- Multi-task for DMS and 2A3
- Pre-training on RNA sequences
- Ensemble of multiple seeds

### 2nd Place Solution

**Technique**:
- Hybrid CNN-Transformer
- Attention on sequence pairs
- Data augmentation via reverse complement

### 3rd Place Solution

**Innovation**:
- RNA structure features
- Graph neural network on predicted structure
- Physics-informed features

---

## Common Winning Strategies

### 1. RNA Sequence Encoding

```python
import torch
import torch.nn as nn
import numpy as np

def encode_rna_sequence(sequence):
    """One-hot encode RNA sequence"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    indices = [mapping[base] for base in sequence]

    # One-hot encoding
    one_hot = np.zeros((len(sequence), 4))
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1

    return one_hot

def add_positional_features(sequence, max_len=500):
    """Add positional information"""
    seq_len = len(sequence)

    # Positional encoding
    position = np.arange(seq_len) / max_len
    position_sin = np.sin(position * np.pi)
    position_cos = np.cos(position * np.pi)

    # Relative position from ends
    rel_start = np.arange(seq_len) / seq_len
    rel_end = 1 - rel_start

    return np.stack([position_sin, position_cos, rel_start, rel_end], axis=1)
```

### 2. Transformer Model for RNA

```python
import torch
import torch.nn as nn

class RNATransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512):
        super().__init__()

        # Input embedding
        self.embedding = nn.Embedding(4, d_model)  # 4 nucleotides
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads for DMS and 2A3
        self.dms_head = nn.Linear(d_model, 1)
        self.a2_3_head = nn.Linear(d_model, 1)

    def forward(self, x, experiment_type):
        # x: (batch, seq_len) - nucleotide indices
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)

        # Transformer encoding
        features = self.transformer(embedded)

        # Predict reactivity
        if experiment_type == 'DMS_MaP':
            output = self.dms_head(features)
        else:
            output = self.a2_3_head(features)

        return output.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]
```

### 3. CNN-Based Model

```python
class RNACNNModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()

        # Multi-scale convolutions
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)

        # Combine scales
        self.combine = nn.Conv1d(hidden_dim * 3, hidden_dim * 2, kernel_size=1)

        # Deep layers
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim * 2),
            ResidualBlock(hidden_dim * 2),
            ResidualBlock(hidden_dim * 2),
        )

        # Output
        self.output = nn.Conv1d(hidden_dim * 2, 2, kernel_size=1)  # DMS and 2A3

    def forward(self, x):
        # x: (batch, seq_len, 4) -> (batch, 4, seq_len)
        x = x.transpose(1, 2)

        # Multi-scale features
        f1 = F.relu(self.conv1(x))
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))

        # Combine
        combined = torch.cat([f1, f2, f3], dim=1)
        combined = F.relu(self.combine(combined))

        # Deep processing
        features = self.blocks(combined)

        # Output
        output = self.output(features)  # (batch, 2, seq_len)
        return output.transpose(1, 2)  # (batch, seq_len, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
```

### 4. Data Augmentation

```python
def augment_rna(sequence, reactivity):
    """Data augmentation for RNA sequences"""

    augmented = []

    # Reverse complement
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    rev_comp = ''.join(complement[base] for base in reversed(sequence))
    rev_reactivity = reactivity[::-1]
    augmented.append((rev_comp, rev_reactivity))

    # Random noise to reactivity (training only)
    noisy_reactivity = reactivity + np.random.normal(0, 0.1, len(reactivity))
    augmented.append((sequence, noisy_reactivity))

    return augmented

def mask_training(sequence, reactivity, mask_prob=0.15):
    """Mask some positions for regularization"""
    mask = np.random.random(len(sequence)) > mask_prob
    masked_sequence = ['N' if not m else s for s, m in zip(sequence, mask)]
    return ''.join(masked_sequence), reactivity
```

### 5. Loss Function with Signal-to-Noise Weighting

```python
def weighted_mae_loss(y_pred, y_true, signal_to_noise, min_snr=1.0):
    """
    Weight loss by signal-to-noise ratio
    High SNR samples are more reliable
    """
    # Mask invalid positions
    mask = ~torch.isnan(y_true)

    # Calculate per-sample weights based on SNR
    weights = torch.clamp(signal_to_noise, min=min_snr) / min_snr
    weights = weights.unsqueeze(-1).expand_as(y_true)

    # Weighted MAE
    errors = torch.abs(y_pred - y_true)
    weighted_errors = errors * weights * mask.float()

    loss = weighted_errors.sum() / mask.sum()
    return loss

# Alternative: focus on well-measured positions
def filtered_mae_loss(y_pred, y_true, reactivity_error, max_error=0.5):
    """Only compute loss on positions with low measurement error"""
    mask = (reactivity_error < max_error) & ~torch.isnan(y_true)

    errors = torch.abs(y_pred - y_true)
    filtered_errors = errors * mask.float()

    return filtered_errors.sum() / mask.sum()
```

---

## Technical Insights

### RNA Structure and Reactivity

| Position Type | DMS Reactivity | 2A3 Reactivity |
|--------------|----------------|----------------|
| Unpaired | High | High |
| Base-paired | Low | Low |
| Stacked | Low | Low |

### Model Architecture Comparison

| Architecture | Pros | Cons |
|--------------|------|------|
| Transformer | Long-range dependencies | Memory intensive |
| CNN | Fast, local patterns | Limited context |
| Hybrid | Best of both | Complex training |

### Key Features for RNA

| Feature | Description |
|---------|-------------|
| Base identity | A, C, G, U encoding |
| Position | Relative position in sequence |
| Local context | K-mers around position |
| Predicted structure | From tools like ViennaRNA |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460121) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460316) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460403) |
| 4th | [Discussion](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460203) |
| 5th | [Discussion](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460250) |

---

## Citation

```bibtex
@misc{stanford-ribonanza-rna-folding,
    author = {Stanford University},
    title = {Stanford Ribonanza RNA Folding},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/stanford-ribonanza-rna-folding}},
    note = {Kaggle}
}
```
