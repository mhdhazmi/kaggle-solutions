# Yale/UNC-CH - Geophysical Waveform Inversion

> Develop physics-guided machine learning models to solve full-waveform inversion problems

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression / Inverse Problem |
| **Data Domain** | Geophysics / Seismic Data |
| **ML Approach** | Physics-Informed Neural Networks (PINNs) |
| **Key Techniques** | Full-Waveform Inversion, U-Net, CNN |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,365 |
| **Timeline** | 2025 |
| **Evaluation Metric** | Mean Absolute Error (MAE) |
| **Host** | Yale University / UNC Chapel Hill |

## Problem Description

**Full-Waveform Inversion (FWI)** is a fundamental problem in geophysics: given seismic waveform data recorded at the surface, reconstruct the subsurface velocity model.

### The Challenge

- Input: Seismic waveforms (time-series data from sensors)
- Output: Subsurface velocity model (2D/3D image of underground structure)
- Goal: Minimize reconstruction error while respecting physical constraints

### Applications

- Oil and gas exploration
- Earthquake monitoring
- Carbon storage site characterization
- Underground infrastructure mapping

---

## Data Description

### Dataset Structure

| Component | Description |
|-----------|-------------|
| Seismic data | Time-series recordings from multiple sensors |
| Velocity models | Ground truth subsurface velocity fields |
| Source/receiver geometry | Spatial configuration of sensors |

### Physical Setup

```
Surface:    [Sensor 1] [Sensor 2] [Sensor 3] ... [Sensor N]
                ↓           ↓          ↓              ↓
            ╔═══════════════════════════════════════════╗
            ║           Subsurface Velocity            ║
            ║              (to predict)                 ║
            ╚═══════════════════════════════════════════╝
```

---

## Evaluation

**Metric**: Mean Absolute Error (MAE)

```
MAE = (1/N) × Σ |predicted_velocity - true_velocity|
```

Lower MAE indicates better reconstruction of subsurface structure.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |
| ... | ... |

**Total Prize Pool**: $50,000

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Physics-informed neural network architecture
- Encoder-decoder structure (U-Net style)
- Multi-scale feature extraction
- Domain-specific data augmentation

*(Details in linked discussion)*

### 2nd Place Solution

*(Details in linked discussion)*

### 3rd Place Solution

*(Details in linked discussion)*

---

## Common Winning Strategies

### 1. Physics-Informed Architectures
Incorporating wave equation physics into neural network design:
- Physics loss terms
- Wavefield propagation constraints
- Boundary conditions

### 2. Encoder-Decoder Networks
U-Net style architectures for image-to-image translation:
- Seismic gather → Velocity model

### 3. Multi-Scale Processing
Handling different spatial frequencies:
- Coarse-to-fine reconstruction
- Multi-resolution features

### 4. Data Augmentation
- Velocity model perturbations
- Noise addition
- Geometry variations

### 5. Ensemble Methods
Combining multiple architectures for robust predictions.

---

## Technical Insights

### Physics of FWI

The wave equation relates seismic data to velocity:

```
∂²u/∂t² = v² × ∇²u + s(t)
```

Where:
- `u` = wavefield
- `v` = velocity (what we want to predict)
- `s(t)` = source function

### Traditional vs ML Approach

| Traditional FWI | ML-Based FWI |
|-----------------|--------------|
| Iterative optimization | Direct prediction |
| Computationally expensive | Fast inference |
| Physics-exact | Data-driven approximation |
| Requires good initial model | End-to-end learning |

### Key Challenges

1. **Ill-posedness**: Multiple velocity models can produce similar data
2. **Computational cost**: Traditional methods require many forward simulations
3. **Noise sensitivity**: Seismic data is inherently noisy
4. **Generalization**: Models trained on synthetic data must work on real data

---

## Technical Requirements

| Requirement | Value |
|-------------|-------|
| Runtime | Standard Kaggle notebook |
| Internet | Allowed |
| External Data | Allowed |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/waveform-inversion/discussion/587388) |
| 2nd | [Discussion](https://www.kaggle.com/c/waveform-inversion/discussion/587950) |
| 3rd | [Discussion](https://www.kaggle.com/c/waveform-inversion/discussion/587419) |
| 4th | [Discussion](https://www.kaggle.com/c/waveform-inversion/discussion/587500) |
| 5th | [Discussion](https://www.kaggle.com/c/waveform-inversion/discussion/587443) |

---

## Citation

```bibtex
@misc{waveform-inversion,
    title = {Yale/UNC-CH - Geophysical Waveform Inversion},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/waveform-inversion}},
    note = {Kaggle, Yale University, UNC Chapel Hill}
}
```
