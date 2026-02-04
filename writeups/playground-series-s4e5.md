# Playground Series - Season 4, Episode 5

> Regression with a Flood Prediction Dataset

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / Environmental / Disaster Risk |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Target Transformation |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,788 |
| **Timeline** | May 2024 |
| **Evaluation Metric** | R² Score |
| **Host** | Kaggle |

## Problem Description

Predict flood probability based on environmental and geographic features, a practical problem for disaster preparedness and risk assessment.

### The Task

- Regression: predict continuous flood probability
- Use environmental and geographic features
- Handle synthetic data based on flood modeling
- Practical application for disaster planning

### Dataset Origin

Synthetically generated from a deep learning model trained on flood prediction data.

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with FloodProbability target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Features

| Category | Examples |
|----------|----------|
| Geography | Elevation, slope, drainage |
| Precipitation | Rainfall intensity, duration |
| Land use | Urban, forest, agricultural |
| Infrastructure | Drainage capacity |
| Historical | Past flood events |

---

## Evaluation

**Metric**: R² Score (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)
```

Higher is better. R² = 1 means perfect prediction.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Gradient boosting ensemble
- Careful feature engineering
- Cross-validation strategy

### 2nd Place Solution

**Technique**:
- CatBoost and LightGBM ensemble
- Feature interactions
- Target distribution analysis

### 4th Place Solution

**Innovation**:
- Original data integration
- Extensive EDA
- Simple but effective features

---

## Common Winning Strategies

### 1. Feature Engineering

```python
# Geographic features
df['elevation_slope_ratio'] = df['elevation'] / (df['slope'] + 1)
df['drainage_capacity'] = df['drainage_density'] * df['soil_permeability']

# Precipitation features
df['rainfall_intensity'] = df['max_rainfall'] / df['duration']
df['cumulative_risk'] = df['annual_rainfall'] * df['drainage_coefficient']

# Land use interactions
df['urban_runoff'] = df['urban_area'] * (1 - df['permeability'])
df['vegetation_protection'] = df['forest_cover'] * df['canopy_interception']
```

### 2. Gradient Boosting Configuration

```python
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# LightGBM
lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

# CatBoost
catboost = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=100
)
```

### 3. Cross-Validation for R²

```python
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

def cv_r2(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        oof_preds[val_idx] = preds
        r2 = r2_score(y_val, preds)
        r2_scores.append(r2)

    return np.mean(r2_scores), oof_preds
```

### 4. Ensemble Strategy

```python
# Simple weighted average
def ensemble_predict(models, weights, X):
    predictions = np.zeros(len(X))
    for model, weight in zip(models, weights):
        predictions += weight * model.predict(X)
    return predictions

# Typical weights
weights = [0.4, 0.35, 0.25]  # LGB, CatBoost, XGBoost
```

### 5. Target Analysis

```python
# Analyze target distribution
print(f"Target mean: {y.mean():.4f}")
print(f"Target std: {y.std():.4f}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

# Check for transformation needs
# If right-skewed, consider log transform
if y.skew() > 1:
    y_transformed = np.log1p(y)
```

---

## Technical Insights

### Flood Prediction Factors

| Factor | Impact on Flood Risk |
|--------|---------------------|
| Elevation | Lower = higher risk |
| Slope | Lower = slower drainage |
| Rainfall | Higher = higher risk |
| Urbanization | Increases runoff |
| Drainage | Better = lower risk |
| Soil | Clay = higher risk |

### R² Interpretation

| R² Value | Interpretation |
|----------|----------------|
| > 0.9 | Excellent |
| 0.7 - 0.9 | Good |
| 0.5 - 0.7 | Moderate |
| < 0.5 | Weak |

### Common Pitfalls

1. **Overfitting**: Complex models may overfit synthetic data
2. **Feature leakage**: Target-related features
3. **Distribution shift**: Synthetic vs real patterns
4. **CV-LB gap**: Trust CV over public leaderboard

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s4e5/discussion/509043) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s4e5/discussion/509410) |
| 4th | [Discussion](https://www.kaggle.com/c/playground-series-s4e5/discussion/509044) |

---

## Citation

```bibtex
@misc{playground-series-s4e5,
    author = {Walter Reade and Elizabeth Park},
    title = {Regression with a Flood Prediction Dataset},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e5}},
    note = {Kaggle Playground Series}
}
```
