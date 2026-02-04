# NeurIPS - Ariel Data Challenge 2024

> Derive exoplanet signals from Ariel's optical instruments

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Signal Extraction / Regression |
| **Data Domain** | Astronomy / Exoplanet Science |
| **ML Approach** | Bayesian Inference + Deep Learning |
| **Key Techniques** | Noise Modeling, Time Series Analysis, Physical Priors |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,151 |
| **Timeline** | August - November 2024 |
| **Evaluation Metric** | Ariel Gaussian Log Likelihood |
| **Host** | University College London / NeurIPS |

## Problem Description

Extract faint exoplanetary atmospheric signals from simulated observations of the upcoming ESA Ariel Mission, dealing with complex noise from spacecraft jitter and other sources.

### The Challenge

- Extract chemical spectra from exoplanet atmospheres
- Handle extremely faint signals (50-200 parts per million)
- Deal with "jitter noise" from spacecraft vibration
- Process multi-wavelength spectroscopic data

### Why It Matters

- **Exoplanet Science**: Characterize atmospheres of distant worlds
- **ESA Ariel Mission**: Preparing for 2029 launch
- **Noise Modeling**: Spacecraft stabilization is imperfect
- **Spectroscopy**: Understand chemical compositions

---

## Data Description

### The Ariel Mission

The European Space Agency's ARIEL mission will study ~1,000 exoplanet atmospheres by observing planets as they transit their host stars.

### Signal Characteristics

| Property | Value |
|----------|-------|
| Signal strength | 50-200 ppm |
| Noise level | ~200 ppm (jitter alone) |
| Challenge | Signal comparable to noise |
| Wavelengths | Multiple infrared channels |

### Files

| File | Description |
|------|-------------|
| `train/test_adc_info.csv` | ADC conversion parameters |
| Signal files | Time series observations |
| Calibration data | Instrument characteristics |

### Data Challenges

- Extremely low signal-to-noise ratio
- Jitter noise from spacecraft pointing instability
- Multiple noise sources (detector, photon noise)
- Real exoplanet cases in test set (not scored)

---

## Evaluation

**Metric**: Ariel Gaussian Log Likelihood

Custom metric measuring agreement between predicted spectra and ground truth, accounting for uncertainties.

### Hidden Test Set

- ~800 exoplanets in hidden test
- Some based on real exoplanets (ignored for scoring)
- Must generalize to unseen planetary systems

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Author**: daiwakun

**Key Approach**:
- Deep learning-based signal extraction
- Careful noise modeling
- Data augmentation strategies
- Ensemble of neural networks

### 2nd Place Solution

**Author**: Jeroen Cottaar

**Technique**: Pure Bayesian Inference, No Deep Learning
- Full Bayesian treatment of the problem
- Physical model of the instrument
- MCMC sampling for posterior
- No neural networks used

This solution was notable for demonstrating that classical statistical methods could compete with deep learning.

### 3rd Place Solution

**Key Approach**:
- Hybrid physical-ML model
- Transit modeling
- Noise decorrelation

### 4th Place Solution

**Author**: greySnow

**Innovation**:
- Signal processing techniques
- Physical constraints
- Efficient inference

---

## Common Winning Strategies

### 1. Understanding the Physical Problem

Transit spectroscopy fundamentals:
```python
# During transit, starlight filters through planet atmosphere
# Different wavelengths absorbed by different molecules
# Transit depth varies with wavelength -> spectrum

def transit_depth(wavelength, atmosphere):
    # Rp/Rs ratio varies with wavelength
    base_depth = (R_planet / R_star) ** 2
    atmosphere_contribution = atmosphere.absorption(wavelength)
    return base_depth + atmosphere_contribution
```

### 2. Jitter Noise Modeling

```python
# Jitter causes pointing variations
# Results in PSF (point spread function) variations
# Must model and remove this noise

def model_jitter_noise(pointing_data, flux_data):
    # Correlate flux variations with pointing
    coefficients = fit_pointing_model(pointing_data, flux_data)
    jitter_component = apply_model(pointing_data, coefficients)
    corrected_flux = flux_data - jitter_component
    return corrected_flux
```

### 3. Bayesian Approach (2nd Place)

```python
import pymc as pm

with pm.Model() as model:
    # Prior on atmospheric spectrum
    spectrum = pm.Normal('spectrum', mu=prior_mean, sigma=prior_std, shape=n_wavelengths)

    # Physical transit model
    expected_signal = transit_model(spectrum)

    # Noise model
    noise_scale = pm.HalfNormal('noise', sigma=1)

    # Likelihood
    likelihood = pm.Normal('obs', mu=expected_signal, sigma=noise_scale, observed=data)

    # Sample posterior
    trace = pm.sample(1000, tune=500)
```

### 4. Data Augmentation

```python
# Generate synthetic observations with known spectra
def augment_data(real_data, n_augmentations):
    augmented = []
    for _ in range(n_augmentations):
        # Vary noise parameters
        # Vary stellar parameters
        # Keep physical constraints
        synthetic = generate_synthetic_observation(real_data)
        augmented.append(synthetic)
    return augmented
```

### 5. Multi-Channel Processing

| Channel | Wavelength | Information |
|---------|------------|-------------|
| VISPhot | Visible | Stellar activity |
| FGS1/2 | Near-IR | Photometry |
| NIRSpec | 1.1-1.95 μm | Spectroscopy |
| AIRS | 1.95-7.8 μm | Full spectrum |

---

## Technical Insights

### The Signal Extraction Problem

| Challenge | Impact |
|-----------|--------|
| Low SNR | Signal ~50-200 ppm, noise ~200 ppm |
| Systematics | Correlated noise from spacecraft |
| Degeneracies | Multiple physical effects |
| Calibration | Instrument response varies |

### Classical vs ML Approaches

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| Bayesian | Uncertainty quantification, interpretable | Computationally expensive |
| Deep Learning | Pattern recognition, fast inference | Black box, needs lots of data |
| Hybrid | Best of both worlds | Complex to implement |

### Key Physics

```
Transit Signal Components:
1. Stellar baseline flux
2. Transit light curve (planet blocks starlight)
3. Limb darkening (star brighter in center)
4. Atmospheric absorption (wavelength-dependent depth)
5. Jitter-induced variations
6. Detector noise
```

### Importance of Priors

Physical priors help constrain solutions:
- Atmospheric spectra must be physically plausible
- Molecular features have known shapes
- Transit depths are bounded
- Noise characteristics are partly known

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | Code competition limits |
| Internet | Disabled |
| GPU | Available |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/544317) |
| 2nd | [Discussion](https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543853) |
| 3rd | [Discussion](https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543944) |
| 4th | [Discussion](https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/544471) |
| 5th | [Discussion](https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543760) |

---

## Resources

- [ESA Ariel Mission](https://arielmission.space/)
- [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

## Citation

```bibtex
@misc{ariel-data-challenge-2024,
    author = {University College London},
    title = {NeurIPS - Ariel Data Challenge 2024},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/ariel-data-challenge-2024}},
    note = {NeurIPS, Kaggle}
}
```
