# Playground Series - Season 5, Episode 2

> Backpack Prediction Challenge

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / E-commerce / Product |
| **ML Approach** | Gradient Boosting Ensemble |
| **Key Techniques** | Feature Engineering, Stacking, Cross-validation |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,393 |
| **Timeline** | February 2025 |
| **Evaluation Metric** | Mean Squared Error (MSE) |
| **Host** | Kaggle |

## Problem Description

Predict a target variable related to backpack characteristics or pricing.

### The Task

Given features about backpacks (likely including materials, size, brand, features), predict a continuous target variable.

---

## Data Description

### Dataset Overview

The dataset was synthetically generated from a deep learning model trained on real backpack-related data.

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Expected Features

Typical backpack dataset features might include:
- **Size**: Dimensions, volume capacity
- **Material**: Fabric type, durability
- **Brand**: Manufacturer
- **Features**: Pockets, straps, waterproofing
- **Weight**: Empty weight
- **Price factors**: Premium features, market segment

---

## Evaluation

**Metric**: Mean Squared Error (MSE)

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

Lower is better.

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
- Careful hyperparameter tuning

### 2nd Place Solution

**Technique**:
- LightGBM + XGBoost blend
- Target encoding for categoricals
- Feature interactions

### 3rd Place Solution

**Innovation**:
- Neural network in ensemble
- Dimensionality reduction features
- Robust cross-validation

---

## Common Winning Strategies

### 1. Feature Engineering

| Feature Type | Examples |
|--------------|----------|
| Ratios | Volume/Weight ratio |
| Interactions | Material × Size |
| Aggregations | Mean target by category |
| Binning | Size categories |

### 2. Model Ensemble

```python
final = 0.4 * lgbm + 0.35 * xgb + 0.25 * catboost
```

### 3. Cross-Validation

- Standard K-Fold (k=5 or k=10)
- Stratified if target has natural groups
- Repeated for stability

### 4. Hyperparameter Tuning

Key parameters:
- Learning rate
- Max depth
- Number of estimators
- Regularization terms

---

## Technical Insights

### MSE Optimization

MSE directly optimized by most models. Consider:
- Outlier impact (MSE sensitive to outliers)
- Target transformation if skewed
- Ensemble averaging reduces variance

### Feature Importance

Identify key predictors:
```python
importance = model.feature_importances_
top_features = sorted(zip(features, importance), key=lambda x: -x[1])
```

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s5e2/discussion/565539) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s5e2/discussion/565542) |
| 3rd | [Discussion](https://www.kaggle.com/c/playground-series-s5e2/discussion/565653) |

---

## Citation

```bibtex
@misc{playground-series-s5e2,
    author = {Walter Reade and Elizabeth Park},
    title = {Backpack Prediction Challenge},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e2}},
    note = {Kaggle Playground Series}
}
```
