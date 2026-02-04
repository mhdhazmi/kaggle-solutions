# NeurIPS - Ariel Data Challenge 2025

> Derive exoplanet signals from Ariel's optical instruments

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression / Signal Processing / Uncertainty Quantification |
| **Data Domain** | Astrophysics / Exoplanet Spectroscopy |
| **ML Approach** | Bayesian Inference (No Neural Networks!) |
| **Key Techniques** | Transit Analysis, Gaussian Processes, Jitter Correction |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition (NeurIPS Track) |
| **Total Prize** | $50,000 |
| **Timeline** | June 26, 2025 - September 25, 2025 |
| **Evaluation Metric** | Gaussian Log-Likelihood (GLL) |
| **Host** | University College London / ESA |

## Problem Description

When an exoplanet passes in front of its host star, a small amount of starlight filters through the planet's atmosphere. This technique, known as **transit spectroscopy**, enables researchers to analyze atmospheric composition. However, these signals are incredibly faint, hidden by complex time-dependent noise.

### The Challenge

Extract the true exoplanet spectrum (transit depth as function of wavelength) from noisy observations, including **confidence intervals**.

### ESA's Ariel Mission

- **Launch**: 2029
- **Goal**: Characterize atmospheres of ~1,000 exoplanets
- **Impact**: Understanding distant worlds, searching for biosignatures

### 2025 vs 2024 Competition

More realistic dataset this year:
- Stellar limb darkening
- Validated star-planet pairs
- More diverse atmospheric models
- Ariel's actual observation cadence
- Repeated observations of some planets

---

## Data Description

### Instruments

| Instrument | Description | Wavelength Range |
|------------|-------------|------------------|
| **FGS1** | Fine Guidance System (photometry) | 0.60 - 0.80 µm |
| **AIRS-CH0** | InfraRed Spectrometer | 1.95 - 3.90 µm |

### Dataset Files

| File | Description |
|------|-------------|
| `train.csv` | Ground truth spectra |
| `wavelengths.csv` | Wavelength grid for spectra |
| `axis_info.parquet` | Axis information for both instruments |
| `adc_info.csv` | Analog-to-digital conversion parameters |
| `[train/test]_star_info.csv` | Star-planet system parameters |

### Star-Planet Parameters

| Parameter | Description |
|-----------|-------------|
| `Rs` | Stellar radius (solar radii) |
| `Ms` | Stellar mass (solar masses) |
| `Ts` | Stellar temperature (Kelvin) |
| `Mp` | Planetary mass (Jupiter masses) |
| `e` | Orbital eccentricity |
| `P` | Orbital period (days) |
| `sma` | Semi-major axis (stellar radii) |
| `i` | Orbital inclination (degrees) |

### Test Set

- ~1,100 exoplanets
- Hidden test set (served at submission time)

---

## Evaluation

**Metric**: Gaussian Log-Likelihood (GLL)

```
GLL = -0.5 × (log(2π) + log(σ²_user) + (y - μ_user)² / σ²_user)
```

### Score Calculation

```
score = (L - L_ref) / (L_ideal - L_ref)
```

- **L_ideal**: Perfect predictions with 10 ppm uncertainty (AIRS) / 1 ppm (FGS1)
- **L_ref**: Training set mean/variance as baseline

### Channel Weights

| Channel | Weight |
|---------|--------|
| FGS1 | 57.846 (emphasized) |
| AIRS-CH0 | 1.0 per spectral point |

**Score Range**: [0, 1] (higher = better)

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |
| 6th | $5,000 |

---

## Top Solutions

### 1st Place: Bayesian Inference Approach

**Author**: jcottaar

*"No neural networks here..."*

*"The Bayesian approach is not used nearly as often as it should be these days"*

#### Three-Step Pipeline

```
Raw Data → Preprocessing → Bayesian Inference → Fudging → Final Predictions
```

**1. Preprocessing**: Convert raw counts to photon counts per pixel

**2. Bayesian Inference**: Principled statistical approach

**3. Fudging**: Apply empirical corrections to optimize score

#### 1. Preprocessing

**Key Challenge**: The infamous **jitter** (pointing error and defocus).

**Jitter Detection**: PCA over all AIRS frames reveals jitter shapes.

**Effects to Handle**:
- Invalid pixels
- Non-Poisson noise (lower for brighter pixels)
- Jitter shapes not summing exactly to zero
- Background signal removal

**Time Binning**: 5 frames (AIRS), 50 frames (FGS1)

#### 2. Bayesian Inference

**Core Concept**:
- **Prior**: Statistical belief about reality (physics model)
- **Observations**: Measured signals
- **Posterior**: Updated belief (transit depth + uncertainty)

**Signal Decomposition** (4 components):

| Component | Description |
|-----------|-------------|
| **Noise** | Uncorrelated Gaussian per time/wavelength |
| **Star spectrum** | Value per wavelength |
| **Drift** | Third-order polynomial over time/wavelength |
| **Transit window** | Batman package modeling |

**Transit Parameters**:
- Transit mid-time t₀ (separate for FGS/AIRS)
- Semi-major axis (sma)
- Period (P)
- Orbital inclination (i)
- Quadratic limb darkening (u₀, u₁)
- Transit depth (Rp²/Rs²)

**Transit Depth Modeling**:
- Mean: Single Gaussian value
- Variation FGS: Single Gaussian
- Variation AIRS: Gaussian Process (two squared-exponential kernels)
- Variation PCA: Fixed basis functions from training data

#### Inference Solver

**Challenge**: Non-linear relationship between prior and observations (batman modeling, multiplicative effects).

**Solution**: Iterative linearization around posterior mean.

**Process**:
1. **Grid search**: Sweep transit depth and mid-time
2. **BFGS optimization**: Fit drift and transit parameters
3. **Full Bayesian inference**: 4-8 iterations of non-linear solver

**Output**: Transit depth prediction + covariance matrix (200 samples)

#### Key Insight

> "My approach isn't actually working..."

Despite principled approach, significant empirical "fudging" was needed to optimize the competition metric.

---

## Common Winning Strategies

### 1. Bayesian Inference
Principled statistical approach for uncertainty quantification.

### 2. Physics-Based Modeling
Incorporating domain knowledge (transit physics, limb darkening).

### 3. Jitter Correction
PCA-based identification and removal of pointing errors.

### 4. Gaussian Processes
Modeling wavelength-dependent transit depth variations.

### 5. Iterative Refinement
Non-linear solver with repeated linearization.

---

## Technical Insights

### Why Bayesian Inference Works

1. **Uncertainty quantification**: Competition requires confidence intervals
2. **Physical priors**: Domain knowledge improves predictions
3. **Principled**: Combines prior beliefs with observations optimally

### Challenge: Very Low SNR

- Tiny planet obscuring huge star
- Multiple noise sources (instrument, stellar variability)
- Requires careful signal extraction

### Transit Spectroscopy Physics

```
Transit Depth ∝ (Rp/Rs)² × atmospheric absorption(λ)
```

Different wavelengths probe different atmospheric layers and molecules.

---

## Technical Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | ≤9 hours |
| GPU Runtime | ≤9 hours |
| Internet Access | Disabled |
| External Data | Allowed |

---

## Solution Links

| Place | Author | Solution |
|-------|--------|----------|
| 1st | jcottaar | [Writeup](https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/1st-place-solution-bayesian-inference-of-course) |
| 2nd | - | [Writeup](https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/2nd-place-solution) |

---

## Resources

- [Ariel Red Book](https://arielmission.space/) - Mission documentation
- [Batman Package](https://github.com/lkreidberg/batman) - Transit modeling
- [Competition Code](https://github.com/jcottaar/ariel2) - 1st place full development history

---

## Citation

```bibtex
@misc{ariel-data-challenge-2025,
    author = {K.H. Yip and L.V. Mugnai and R.L. Coates and others},
    title = {NeurIPS - Ariel Data Challenge 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/ariel-data-challenge-2025}},
    note = {Kaggle, University College London, NeurIPS}
}
```
