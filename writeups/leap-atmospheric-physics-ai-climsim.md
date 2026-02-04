# LEAP - Atmospheric Physics using AI (ClimSim)

> Simulate higher resolution atmospheric processes within E3SM-MMF, a climate model supported by the U.S. Department of Energy

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Output Regression |
| **Data Domain** | Climate Science / Atmospheric Physics |
| **ML Approach** | Neural Networks + Physical Constraints |
| **Key Techniques** | Physics-Informed ML, Multi-Target Regression |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Prediction Competition |
| **Total Prize** | $50,000 |
| **Teams** | 693 |
| **Timeline** | April - July 2024 |
| **Evaluation Metric** | R² Score |
| **Host** | LEAP (Learning the Earth with AI and Physics) |

## Problem Description

Develop ML models to emulate subgrid-scale atmospheric physics in the E3SM-MMF climate model, replacing expensive physical simulations with fast ML inference.

### The Challenge

- Emulate storms, clouds, turbulence, rainfall, and radiation
- Replace Multi-scale Modeling Framework (MMF) with ML
- Maintain physical consistency
- Enable affordable high-resolution climate projections

### Why It Matters

- **Climate Science**: Better understanding of future climate
- **Computational Cost**: MMF is too expensive for operational use
- **Uncertainty Reduction**: Improve climate projection accuracy
- **Accessibility**: Make high-res projections broadly available

---

## Data Description

### Climate Model Context

The E3SM-MMF model uses a Multi-scale Modeling Framework to explicitly represent subgrid processes that typical climate models must approximate (parameterize).

### Input Variables

| Category | Examples |
|----------|----------|
| Temperature | Atmospheric temperature profiles |
| Humidity | Specific humidity at levels |
| Wind | U, V components |
| Pressure | Surface and level pressures |
| Radiation | Solar, longwave fluxes |

### Output Variables (Targets)

Tendencies (rates of change) for:
- Temperature
- Moisture
- Cloud properties
- Precipitation
- Radiation fluxes

### Data Scale

| Property | Value |
|----------|-------|
| Vertical levels | ~60 atmospheric layers |
| Time steps | Multiple years of simulation |
| Grid points | Global coverage |

---

## Evaluation

**Metric**: R² Score (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)
```

Where:
- SS_res = Σ(y - ŷ)²
- SS_tot = Σ(y - ȳ)²

Higher is better. R² = 1 means perfect prediction.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Deep neural network architecture
- Physical constraints in loss function
- Careful normalization
- Ensemble methods

### 2nd Place Solution

**Technique**:
- U-Net style architecture
- Attention mechanisms
- Multi-scale features

### 3rd Place Solution

**Innovation**:
- Residual connections
- Layer-wise predictions
- Physical consistency checks

---

## Common Winning Strategies

### 1. The Secret for Beating Baseline

Key insight from competition discussions:
```python
# Proper normalization is crucial
# Climate variables span very different scales

def normalize_inputs(X, means, stds):
    return (X - means) / (stds + 1e-8)

def denormalize_outputs(Y, means, stds):
    return Y * stds + means
```

### 2. Neural Network Architecture

```python
import torch.nn as nn

class ClimateEmulator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

### 3. Physics-Informed Loss

```python
def physics_loss(predictions, targets, physical_constraints):
    # Standard MSE loss
    mse_loss = F.mse_loss(predictions, targets)

    # Energy conservation
    energy_pred = compute_energy(predictions)
    energy_target = compute_energy(targets)
    energy_loss = F.mse_loss(energy_pred, energy_target)

    # Mass conservation
    mass_pred = compute_mass(predictions)
    mass_target = compute_mass(targets)
    mass_loss = F.mse_loss(mass_pred, mass_target)

    return mse_loss + 0.1 * energy_loss + 0.1 * mass_loss
```

### 4. Multi-Output Handling

```python
# Multiple output variables with different scales
class MultiHeadOutput(nn.Module):
    def __init__(self, shared_dim, output_dims):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(shared_dim, dim)
            for name, dim in output_dims.items()
        })

    def forward(self, shared_features):
        return {name: head(shared_features) for name, head in self.heads.items()}
```

### 5. Vertical Level Processing

```python
# Atmospheric data has vertical structure
# Process levels together or separately

# Option 1: Flatten all levels
x_flat = x.reshape(batch_size, -1)

# Option 2: 1D convolution over levels
conv = nn.Conv1d(n_features, hidden, kernel_size=3, padding=1)
x_conv = conv(x.transpose(1, 2))

# Option 3: Transformer for level interactions
transformer = nn.TransformerEncoder(...)
x_trans = transformer(x)
```

---

## Technical Insights

### Why ML for Climate Models?

| Aspect | Traditional Parameterization | ML Emulator |
|--------|------------------------------|-------------|
| Accuracy | Limited | Potentially higher |
| Speed | Slow (MMF) | Very fast |
| Flexibility | Fixed equations | Learns from data |
| Physical consistency | Guaranteed | Must be enforced |

### Subgrid Processes

The model must learn to predict:
- **Convection**: Vertical air movement
- **Cloud microphysics**: Droplet formation
- **Turbulence**: Mixing processes
- **Radiation**: Energy transfer
- **Precipitation**: Rain, snow formation

### Challenges in Climate ML

| Challenge | Approach |
|-----------|----------|
| Distribution shift | Robust training |
| Physical constraints | Physics-informed loss |
| Long-term stability | Careful validation |
| Extreme events | Balanced sampling |

### The Competition's Scientific Impact

From organizer discussion: "This competition has officially pushed science forward" - results contributed to published research on ML-based climate model parameterization.

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523063) |
| 2nd | [Discussion](https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523055) |
| 3rd | [Discussion](https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523077) |
| 7th | [Discussion](https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/524111) |
| 8th | [Discussion](https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523223) |

---

## Resources

- [ClimSim Dataset](https://huggingface.co/datasets/LEAP/ClimSim)
- [E3SM Model](https://e3sm.org/)

---

## Citation

```bibtex
@misc{leap-atmospheric-physics-ai-climsim,
    author = {LEAP},
    title = {LEAP - Atmospheric Physics using AI (ClimSim)},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/leap-atmospheric-physics-ai-climsim}},
    note = {Kaggle}
}
```
