# Playground Series - Season 5, Episode 3

> Binary Prediction with a Rainfall Dataset

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification |
| **Data Domain** | Tabular / Weather / Climate |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Time Series Features, Ensemble |
| **Difficulty** | Beginner |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 4,381 |
| **Participants** | 4,545 |
| **Timeline** | March 2025 |
| **Evaluation Metric** | ROC AUC Score |
| **Host** | Kaggle |

## Problem Description

Predict whether it will rain on a given day based on weather features.

### The Task

Binary classification:
- **0**: No rainfall
- **1**: Rainfall occurred

Given weather measurements and atmospheric conditions, predict the probability of rainfall.

### Why It Matters

- **Agriculture**: Crop planning and irrigation decisions
- **Events**: Outdoor activity planning
- **Infrastructure**: Flood preparation
- **Aviation**: Flight operations

---

## Data Description

### Dataset Overview

The dataset was synthetically generated from a deep learning model trained on the "Rainfall Prediction using Machine Learning" dataset.

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with rainfall target (binary) |
| `test.csv` | Test data for probability predictions |
| `sample_submission.csv` | Example submission format |

### Expected Features

Typical rainfall prediction features:

| Feature | Description |
|---------|-------------|
| Temperature | Max, min, average temperatures |
| Humidity | Relative humidity levels |
| Pressure | Atmospheric pressure |
| Wind | Speed and direction |
| Cloud Cover | Percentage of sky covered |
| Precipitation History | Previous days' rainfall |
| Evaporation | Rate of evaporation |
| Sunshine Hours | Hours of sunlight |

---

## Evaluation

**Metric**: Area Under ROC Curve (AUC)

Measures ability to rank positive examples higher than negative examples.

### AUC Properties

- Range: 0.5 (random) to 1.0 (perfect)
- Threshold-independent
- Handles class imbalance well
- Focuses on ranking, not calibration

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

---

## Top Solutions

### 2nd Place Solution

**Key Approach**:
- Gradient boosting ensemble
- Careful feature engineering
- Weather-specific domain features

### 6th Place Solution

**Technique**:
- LightGBM with tuned hyperparameters
- Feature interactions
- Cross-validation optimization

### 18th Place Solution

**Innovation**:
- Combined original and synthetic data
- Target encoding
- Neural network blend

---

## Common Winning Strategies

### 1. Feature Engineering

| Feature Type | Examples |
|--------------|----------|
| Lag features | Yesterday's humidity, temp |
| Rolling stats | 3-day moving average |
| Interactions | Humidity × Temperature |
| Differences | Today - Yesterday pressure |
| Date features | Month, season |

### 2. Weather Domain Features

```python
# Dew point approximation
dew_point = temp - ((100 - humidity) / 5)

# Pressure tendency
pressure_change = pressure_today - pressure_yesterday

# Heat index
heat_index = -42.379 + 2.04901523*T + 10.14333127*RH - ...
```

### 3. Model Ensemble

Typical winning blend:
```python
final = 0.4 * lgbm + 0.35 * xgb + 0.25 * catboost
```

### 4. Threshold Optimization

For AUC, probability calibration helps:
- Platt scaling
- Isotonic regression
- Post-hoc threshold tuning

### 5. Cross-Validation

- Standard K-Fold works well
- Time-based split if temporal data
- Stratified for class balance

---

## Technical Insights

### Weather Prediction Basics

Key meteorological indicators:
1. **Humidity > 70%**: Increased rain likelihood
2. **Falling pressure**: Storm approaching
3. **Wind direction changes**: Front passing
4. **Cloud formation**: Precipitation potential

### Class Imbalance

Rainfall datasets often have imbalance:
- More dry days than rainy days
- Handle with sampling or class weights
- AUC handles this naturally

### Temporal Patterns

Weather has strong temporal structure:
- Seasonal patterns (monsoons, dry seasons)
- Daily cycles (afternoon thunderstorms)
- Multi-day trends (weather systems)

### Feature Correlations

Many weather features are correlated:
- Temperature ↔ Humidity (inverse)
- Pressure ↔ Weather conditions
- Cloud cover ↔ Sunshine hours

Consider dimensionality reduction or feature selection.

---

## Solution Links

| Place | Solution |
|-------|----------|
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s5e3/discussion/571176) |
| 6th | [Discussion](https://www.kaggle.com/c/playground-series-s5e3/discussion/571216) |
| 18th | [Discussion](https://www.kaggle.com/c/playground-series-s5e3/discussion/571021) |
| 37th | [Discussion](https://www.kaggle.com/c/playground-series-s5e3/discussion/571139) |

---

## Citation

```bibtex
@misc{playground-series-s5e3,
    author = {Walter Reade and Elizabeth Park},
    title = {Binary Prediction with a Rainfall Dataset},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e3}},
    note = {Kaggle Playground Series}
}
```
