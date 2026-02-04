# Playground Series - Season 5, Episode 5

> Predict Calorie Expenditure

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / Health & Fitness |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Stacking, Target Transformation |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 4,316 |
| **Timeline** | May 2025 |
| **Evaluation Metric** | Root Mean Squared Logarithmic Error (RMSLE) |
| **Host** | Kaggle |

## Problem Description

Predict how many calories were burned during a workout based on various exercise and physiological features.

### The Task

Given features about a workout session:
- Predict the total calories burned
- Handle the inherent variance in calorie expenditure

### Why RMSLE?

RMSLE is used because:
- Penalizes underestimates more than overestimates (relatively)
- Handles multiplicative errors better
- Works well when target has large range

---

## Data Description

### Dataset Overview

The dataset was synthetically generated from a deep learning model trained on real calorie expenditure data.

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with Calories target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Expected Features

Typical calorie prediction features include:
- **Duration**: Workout length
- **Heart Rate**: Average/max heart rate
- **Age**: Participant age
- **Weight/Height**: Body measurements
- **Gender**: Male/Female
- **Exercise Type**: Activity category
- **Intensity**: Workout intensity level

---

## Evaluation

**Metric**: Root Mean Squared Logarithmic Error (RMSLE)

```
RMSLE = sqrt((1/n) × Σ(log(1 + ŷᵢ) - log(1 + yᵢ))²)
```

Where:
- ŷᵢ = predicted calories
- yᵢ = actual calories
- n = number of samples

### RMSLE Properties

- Penalizes under-prediction more heavily
- Scale-independent (relative errors)
- Target transformation: equivalent to RMSE on log(1+y)

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
- Heavy feature engineering
- Ensemble of gradient boosting models
- Careful handling of log transformation

### 2nd Place Solution

**Technique**:
- LightGBM + XGBoost + CatBoost blend
- Physical-based feature creation
- Cross-validation optimization

### 3rd Place Solution

**Innovation**:
- Neural network combined with GBM
- Residual stacking
- Target encoding for categorical features

---

## Common Winning Strategies

### 1. Target Transformation

For RMSLE optimization:
```python
# Train on log-transformed target
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

# Predict and inverse transform
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
```

### 2. Feature Engineering

| Feature Type | Examples |
|--------------|----------|
| BMI | Weight / Height² |
| Heart Rate Reserve | Max HR - Resting HR |
| MET Estimation | Metabolic equivalent calculation |
| Age Groups | Binned age categories |
| Duration × Intensity | Interaction features |

### 3. Model Ensemble

Typical winning ensemble:
```
Final = 0.4 × LightGBM + 0.3 × XGBoost + 0.2 × CatBoost + 0.1 × Neural Network
```

### 4. Cross-Validation Strategy

- Stratified K-Fold on target bins
- Group K-Fold if user IDs present
- Repeated K-Fold for stability

### 5. Hyperparameter Tuning

Key parameters to tune:
- `learning_rate`: 0.01-0.1
- `max_depth`: 4-10
- `n_estimators`: 500-2000
- `reg_alpha/lambda`: Regularization

---

## Technical Insights

### RMSLE vs RMSE

| Metric | Best For |
|--------|----------|
| RMSE | Absolute errors matter |
| RMSLE | Relative errors matter |
| MAE | Robust to outliers |

### Physics-Based Features

Calorie estimation often uses:
```
Calories ≈ MET × Weight × Duration
```

Where MET (Metabolic Equivalent of Task) varies by activity.

### Handling Zeros

RMSLE uses log(1 + y), so zero values are handled gracefully.

### Negative Predictions

Ensure predictions are non-negative:
```python
y_pred = np.maximum(y_pred, 0)
```

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s5e5/discussion/582611) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s5e5/discussion/582700) |
| 3rd | [Discussion](https://www.kaggle.com/c/playground-series-s5e5/discussion/582848) |
| 4th | [Discussion](https://www.kaggle.com/c/playground-series-s5e5/discussion/582594) |
| 5th | [Discussion](https://www.kaggle.com/c/playground-series-s5e5/discussion/582564) |

---

## Citation

```bibtex
@misc{playground-series-s5e5,
    author = {Walter Reade and Elizabeth Park},
    title = {Predict Calorie Expenditure},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e5}},
    note = {Kaggle Playground Series}
}
```
