# Playground Series - Season 4, Episode 9

> Regression of Used Car Prices

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression |
| **Data Domain** | Tabular / Automotive / E-commerce |
| **ML Approach** | Neural Networks + Gradient Boosting |
| **Key Techniques** | Feature Engineering, Stacking, Blending |
| **Difficulty** | Beginner-Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Playground Prediction Competition |
| **Prize** | Kaggle Swag |
| **Teams** | 3,066 |
| **Timeline** | September 2024 |
| **Evaluation Metric** | Root Mean Squared Error (RMSE) |
| **Host** | Kaggle |

## Problem Description

Predict the price of used cars based on various vehicle attributes like make, model, year, mileage, and condition.

### The Task

- Regression problem: predict continuous price values
- Handle categorical features (make, model, fuel type)
- Account for non-linear depreciation patterns
- Classic regression problem with real-world applicability

### Dataset Origin

The dataset was synthetically generated from a deep learning model trained on the "Used Car Price Prediction Dataset". Feature distributions are close to but not exactly the same as the original.

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with price target |
| `test.csv` | Test data for predictions |
| `sample_submission.csv` | Example submission format |

### Features (27 columns)

| Category | Examples |
|----------|----------|
| Vehicle ID | Brand, model, variant |
| Specifications | Engine size, fuel type, transmission |
| Condition | Year, mileage, owner history |
| Features | Seats, color |
| Location | Seller location |

---

## Evaluation

**Metric**: Root Mean Squared Error (RMSE)

```
RMSE = sqrt((1/N) * Σ(yᵢ - ŷᵢ)²)
```

Lower is better. Penalizes large errors heavily.

### Submission Format

```
id,price
188533,43878.016
188534,43878.016
188535,43878.016
```

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

Part of the **AutoML Grand Prix** series.

---

## Top Solutions

### 1st Place Solution

**Author**: Mart Preusse

**Key Approach**: Stacked Neural Networks
- Multiple neural network architectures
- Stacking ensemble
- Careful feature preprocessing
- Target transformation

### 3rd Place Solution

**Technique**: "Gather, Ridge, ..."
- Feature gathering from multiple sources
- Ridge regression in ensemble
- Diverse base models

### 4th Place Solution

**Author**: Tilii

**Innovation**: "Blending works, but 'blending' doesn't"
- Distinction between true blending vs naive averaging
- Careful validation strategy
- Optimized blend weights

---

## Common Winning Strategies

### 1. Target Transformation

For price prediction, log transform helps:
```python
# Log transform target
y_train_log = np.log1p(y_train)

# Train on log-transformed target
model.fit(X_train, y_train_log)

# Inverse transform predictions
y_pred = np.expm1(model.predict(X_test))
```

### 2. Feature Engineering

```python
# Age-related features
df['vehicle_age'] = 2024 - df['year']
df['age_squared'] = df['vehicle_age'] ** 2

# Mileage per year
df['mileage_per_year'] = df['km_driven'] / (df['vehicle_age'] + 1)

# Brand encoding based on average price
brand_price_mean = df.groupby('brand')['price'].mean()
df['brand_price_rank'] = df['brand'].map(brand_price_mean)

# Categorical combinations
df['brand_model'] = df['brand'] + '_' + df['model']
df['fuel_transmission'] = df['fuel_type'] + '_' + df['transmission']
```

### 3. Stacked Neural Networks (1st Place)

```python
# Base models
models = [
    create_nn_model(hidden=[256, 128, 64]),
    create_nn_model(hidden=[512, 256]),
    create_nn_model(hidden=[128, 64, 32, 16]),
]

# Train base models and get OOF predictions
oof_predictions = []
for model in models:
    oof = cross_val_predict(model, X_train, y_train, cv=5)
    oof_predictions.append(oof)

# Stack with meta-learner
meta_features = np.column_stack(oof_predictions)
meta_model = Ridge()
meta_model.fit(meta_features, y_train)
```

### 4. Blending Strategy

```python
# True blending requires:
# 1. Diverse base models
# 2. Validation-based weight optimization
# 3. Out-of-fold predictions

from scipy.optimize import minimize

def blend_loss(weights, predictions, y_true):
    blend = sum(w * p for w, p in zip(weights, predictions))
    return np.sqrt(np.mean((blend - y_true) ** 2))

# Optimize weights
result = minimize(
    blend_loss,
    x0=np.ones(n_models) / n_models,
    args=(oof_predictions, y_train),
    method='SLSQP',
    bounds=[(0, 1)] * n_models
)
optimal_weights = result.x
```

### 5. Handling Categorical Features

```python
# Target encoding with smoothing
from category_encoders import TargetEncoder
encoder = TargetEncoder(smoothing=10)

# Frequency encoding
df['brand_count'] = df.groupby('brand')['id'].transform('count')

# Leave-one-out encoding
from category_encoders import LeaveOneOutEncoder
loo_encoder = LeaveOneOutEncoder()
```

---

## Technical Insights

### Car Depreciation Patterns

Price depends non-linearly on:
- **Age**: Steep initial depreciation, then slower
- **Mileage**: Generally linear but brand-dependent
- **Brand**: Premium brands hold value better
- **Fuel type**: Electric vs diesel vs petrol trends

### RMSE vs MAE

| Metric | Behavior |
|--------|----------|
| RMSE | Penalizes large errors more |
| MAE | Equal weight to all errors |

For RMSE optimization, outlier handling is important.

### Neural Networks for Tabular Data

Why NNs worked well here:
1. Large training set
2. Complex feature interactions
3. Embedding layers for categoricals
4. Non-linear price patterns

```python
def create_tabular_nn(cat_dims, num_features):
    # Embeddings for categorical features
    cat_inputs = []
    cat_embeddings = []
    for dim in cat_dims:
        inp = Input(shape=(1,))
        emb = Embedding(dim, min(50, dim//2))(inp)
        emb = Flatten()(emb)
        cat_inputs.append(inp)
        cat_embeddings.append(emb)

    # Numerical features
    num_input = Input(shape=(num_features,))

    # Concatenate all
    concat = Concatenate()(cat_embeddings + [num_input])

    # Dense layers
    x = Dense(256, activation='relu')(concat)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(1)(x)

    return Model(cat_inputs + [num_input], x)
```

### AutoML Grand Prix

This competition was part of Kaggle's AutoML series, encouraging exploration of automated machine learning tools like:
- AutoGluon
- H2O
- FLAML
- Auto-sklearn

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/playground-series-s4e9/discussion/537052) |
| 2nd | [Discussion](https://www.kaggle.com/c/playground-series-s4e9/discussion/537349) |
| 3rd | [Discussion](https://www.kaggle.com/c/playground-series-s4e9/discussion/537029) |
| 4th | [Discussion](https://www.kaggle.com/c/playground-series-s4e9/discussion/536973) |
| 5th | [Discussion](https://www.kaggle.com/c/playground-series-s4e9/discussion/537173) |

---

## Citation

```bibtex
@misc{playground-series-s4e9,
    author = {Walter Reade and Elizabeth Park},
    title = {Regression of Used Car Prices},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s4e9}},
    note = {Kaggle Playground Series}
}
```
