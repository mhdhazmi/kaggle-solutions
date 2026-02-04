# Playground Series - Season 5, Episode 1

> Forecasting Sticker Sales

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Time Series Forecasting / Regression |
| **Data Domain** | Tabular / E-commerce / Retail |
| **ML Approach** | Gradient Boosting + Time Series Features |
| **Key Techniques** | Lag Features, Rolling Statistics, Ensemble |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 2,722 |
| **Timeline** | January 2025 |
| **Evaluation Metric** | Mean Absolute Percentage Error (MAPE) |
| **Host** | Kaggle |

## Problem Description

Forecast future sticker sales based on historical sales data and related features.

### The Task

- Predict sticker sales quantities
- Handle time series patterns
- Account for seasonality and trends

### Why MAPE?

MAPE measures relative prediction error:
- Scale-independent
- Intuitive interpretation (percentage error)
- Penalizes large relative errors

---

## Data Description

### Dataset Overview

The dataset was synthetically generated from a deep learning model trained on real sales data.

### Files

| File | Description |
|------|-------------|
| `train.csv` | Historical sales with target |
| `test.csv` | Future periods for prediction |
| `sample_submission.csv` | Example submission format |

### Expected Features

| Feature Type | Examples |
|--------------|----------|
| Time | Date, day of week, month |
| Product | Sticker type, category |
| Sales history | Past sales, trends |
| External | Holidays, promotions |

---

## Evaluation

**Metric**: Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) × Σ|yᵢ - ŷᵢ| / |yᵢ|
```

Lower is better. Expressed as percentage.

### MAPE Considerations

- Undefined for zero actual values
- Asymmetric (penalizes under-prediction more)
- Good for relative accuracy assessment

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
- Extensive time series feature engineering
- Gradient boosting ensemble
- Careful handling of seasonality

### 2nd Place Solution

**Technique**:
- LightGBM with lag features
- Rolling statistics
- Holiday indicators

### 3rd Place Solution

**Innovation**:
- Hybrid ML + statistical approach
- Feature selection optimization
- Cross-validation on time blocks

---

## Common Winning Strategies

### 1. Time Series Feature Engineering

| Feature | Description |
|---------|-------------|
| Lag features | Sales from t-1, t-7, t-30 |
| Rolling mean | 7-day, 30-day averages |
| Rolling std | Volatility measures |
| Trends | Moving averages of moving averages |
| Seasonal | Day of week, month patterns |

```python
# Lag features
df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)

# Rolling statistics
df['rolling_7_mean'] = df['sales'].rolling(7).mean()
df['rolling_7_std'] = df['sales'].rolling(7).std()
```

### 2. Date Features

```python
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day_of_month'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

### 3. Target Transformation

For MAPE optimization:
```python
# Log transform for better relative error
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)
y_pred = np.expm1(model.predict(X_test))
```

### 4. Cross-Validation Strategy

Time series requires special CV:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # Train on past, validate on future
    pass
```

### 5. Model Ensemble

Combine different approaches:
- LightGBM for tabular patterns
- XGBoost for robustness
- Simple baselines (seasonal naive)

---

## Technical Insights

### Handling Zeros

MAPE is undefined for zero values:
```python
# Option 1: Add small epsilon
mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))

# Option 2: Use SMAPE instead
smape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
```

### Seasonality Patterns

Sticker sales may show:
- Weekly patterns (weekday vs weekend)
- Monthly patterns (beginning/end of month)
- Holiday effects (gift-giving occasions)
- Trend (growing/declining demand)

### Forecast Horizon

For multi-step forecasting:
- Direct method: Separate model per horizon
- Recursive method: Feed predictions back
- Hybrid approaches

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s5e1/discussion/560629) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s5e1/discussion/560549) |
| 3rd | [Discussion](https://www.kaggle.com/c/playground-series-s5e1/discussion/560535) |
| 4th | [Discussion](https://www.kaggle.com/c/playground-series-s5e1/discussion/560692) |
| 5th | [Discussion](https://www.kaggle.com/c/playground-series-s5e1/discussion/560554) |

---

## Citation

```bibtex
@misc{playground-series-s5e1,
    author = {Walter Reade and Elizabeth Park},
    title = {Forecasting Sticker Sales},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e1}},
    note = {Kaggle Playground Series}
}
```
