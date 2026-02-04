# Playground Series - Season 4, Episode 12

> Regression with an Insurance Dataset

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / Insurance / Finance |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Target Transformation, Stacking |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,390 |
| **Timeline** | December 2024 |
| **Evaluation Metric** | Root Mean Squared Logarithmic Error (RMSLE) |
| **Host** | Kaggle |

## Problem Description

Predict insurance-related values (likely premiums or claims) based on policyholder characteristics.

### The Task

Given features about policyholders and their insurance:
- Predict a continuous target (premium, claim amount, etc.)
- Handle the right-skewed nature of insurance data
- Account for various risk factors

---

## Data Description

### Dataset Overview

The dataset was synthetically generated from a deep learning model trained on real insurance data.

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Expected Features

| Category | Features |
|----------|----------|
| Demographics | Age, gender, region |
| Policy | Coverage type, deductible, term |
| Risk factors | Claims history, credit score |
| Vehicle/Property | Age, value, type (if auto/home) |

---

## Evaluation

**Metric**: Root Mean Squared Logarithmic Error (RMSLE)

```
RMSLE = sqrt((1/n) × Σ(log(1 + ŷᵢ) - log(1 + yᵢ))²)
```

### Why RMSLE?

- Insurance values are typically right-skewed
- Relative errors matter more than absolute
- Penalizes under-prediction of large values

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
- Log transformation of target
- Ensemble of gradient boosting models
- Careful feature engineering

### 2nd Place Solution

**Technique**:
- Insurance domain knowledge features
- Risk grouping and encoding
- Neural network in ensemble

### 9th Place Solution

**Innovation**:
- Stacking with diverse base models
- Target encoding with smoothing
- Outlier handling

---

## Common Winning Strategies

### 1. Target Transformation

For RMSLE, train on log-transformed target:
```python
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
```

### 2. Feature Engineering

| Feature | Calculation |
|---------|-------------|
| Age groups | Binned age categories |
| Risk score | Composite of risk factors |
| Premium ratio | Target / coverage amount |
| Interaction | Age × Gender × Region |

### 3. Insurance-Specific Features

```python
# Loss ratio proxy
df['risk_indicator'] = df['claims_count'] / df['policy_age']

# Coverage adequacy
df['coverage_ratio'] = df['coverage_amount'] / df['property_value']
```

### 4. Model Ensemble

```python
final = 0.4 * lgbm + 0.35 * xgb + 0.25 * catboost
```

### 5. Handling Right-Skewed Data

- Log transformation
- Power transformations (Box-Cox, Yeo-Johnson)
- Quantile regression

---

## Technical Insights

### Insurance Data Characteristics

1. **Right-skewed**: Many small values, few large
2. **Zero-inflation**: Many zero claims
3. **Heavy tails**: Extreme values possible
4. **Correlations**: Risk factors interconnected

### RMSLE vs RMSE

| Metric | Sensitivity |
|--------|-------------|
| RMSE | Absolute errors |
| RMSLE | Relative errors (log scale) |
| MAE | Robust to outliers |

### Actuarial Considerations

Traditional insurance pricing uses:
- Generalized Linear Models (GLM)
- Tweedie distributions
- Loss ratio modeling

ML approaches can capture non-linear patterns.

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s4e12/discussion/554328) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s4e12/discussion/554505) |
| 9th | [Discussion](https://www.kaggle.com/c/playground-series-s4e12/discussion/554377) |

---

## Citation

```bibtex
@misc{playground-series-s4e12,
    author = {Walter Reade and Elizabeth Park},
    title = {Regression with an Insurance Dataset},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e12}},
    note = {Kaggle Playground Series}
}
```
