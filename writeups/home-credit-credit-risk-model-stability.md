# Home Credit - Credit Risk Model Stability

> Create a model measured against feature stability over time

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Binary Classification with Stability Constraint |
| **Data Domain** | Finance / Credit Risk |
| **ML Approach** | Gradient Boosting + Stability Optimization |
| **Key Techniques** | Temporal Stability, Feature Engineering, Gini Optimization |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Competition |
| **Total Prize** | $105,000 |
| **Teams** | 3,856 |
| **Timeline** | 2024 |
| **Evaluation Metric** | Gini Stability |
| **Host** | Home Credit |

## Problem Description

Build a credit risk model that not only predicts default well but maintains stable performance across time periods - a critical requirement for real-world deployment.

### The Challenge

- Predict credit default probability
- Model must be stable across different time periods
- Novel metric combining Gini and stability
- Massive dataset with complex features

### Why Stability Matters

Traditional ML competitions optimize for accuracy. In production credit systems:
- Regulations require stable models
- Performance drift causes business issues
- Fairness concerns across time
- Model monitoring is expensive

---

## Data Description

### Dataset Scale

Home Credit's largest dataset:
- Millions of loan applications
- Hundreds of features
- Multiple tables to join
- Time-spanning data

### Data Tables

| Table | Description |
|-------|-------------|
| Base | Main application data |
| Bureau | Credit bureau history |
| Previous applications | Past Home Credit loans |
| Installments | Payment history |
| Credit card | Card balance data |
| POS cash | Point of sale data |

### Temporal Structure

Data spans multiple years with:
- Training period
- Validation periods
- Hidden test periods
- Stability measured across time slices

---

## Evaluation

**Metric**: Gini Stability

```python
def gini_stability(y_true, y_pred, time_periods):
    """
    Combines:
    1. Gini coefficient (discriminative power)
    2. Stability across time periods (variance)
    """
    ginis = []
    for period in time_periods:
        mask = time_periods == period
        gini = 2 * roc_auc_score(y_true[mask], y_pred[mask]) - 1
        ginis.append(gini)

    mean_gini = np.mean(ginis)
    gini_std = np.std(ginis)

    # Penalize high variance
    stability_score = mean_gini - gini_std

    return stability_score
```

Higher is better. Rewards both accuracy AND consistency.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $105,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Focus on feature stability
- Time-based cross-validation
- Careful feature selection
- Ensemble with stability weighting

### 13th Place Solution

**Technique**:
- LightGBM with stability-aware training
- Temporal feature engineering
- Robust feature selection
- Adversarial validation

---

## Common Winning Strategies

### 1. Stability-Aware Cross-Validation

```python
def temporal_cv(df, target_col, date_col, n_splits=5):
    """Time-based CV for stability measurement"""
    dates = df[date_col].sort_values().unique()
    split_size = len(dates) // (n_splits + 1)

    for i in range(n_splits):
        train_end = dates[split_size * (i + 1)]
        val_start = dates[split_size * (i + 1)]
        val_end = dates[split_size * (i + 2)]

        train_mask = df[date_col] <= train_end
        val_mask = (df[date_col] > val_start) & (df[date_col] <= val_end)

        yield df[train_mask], df[val_mask]
```

### 2. Feature Stability Analysis

```python
def analyze_feature_stability(df, feature, date_col, n_periods=12):
    """Check if feature distribution is stable over time"""
    periods = pd.qcut(df[date_col], n_periods, duplicates='drop')

    distributions = []
    for period in periods.unique():
        mask = periods == period
        dist = df.loc[mask, feature].describe()
        distributions.append(dist)

    # Calculate coefficient of variation
    means = [d['mean'] for d in distributions]
    stds = [d['std'] for d in distributions]

    mean_cv = np.std(means) / (np.mean(means) + 1e-8)
    return mean_cv
```

### 3. Stable Feature Selection

```python
def select_stable_features(df, features, target, date_col, stability_threshold=0.3):
    """Select features that have stable predictive power"""
    stable_features = []

    for feature in features:
        # Calculate feature importance across time periods
        importances = []
        for train_df, val_df in temporal_cv(df, target, date_col):
            model = LGBMClassifier()
            model.fit(train_df[[feature]], train_df[target])
            importance = model.feature_importances_[0]
            importances.append(importance)

        # Check stability
        cv = np.std(importances) / (np.mean(importances) + 1e-8)
        if cv < stability_threshold:
            stable_features.append(feature)

    return stable_features
```

### 4. Aggregation Feature Engineering

```python
def create_aggregation_features(df, bureau_df, id_col='SK_ID_CURR'):
    """Create stable aggregate features from bureau data"""
    aggs = {
        'CREDIT_ACTIVE': ['count', 'sum'],
        'AMT_CREDIT_SUM': ['mean', 'sum', 'max'],
        'DAYS_CREDIT': ['mean', 'min', 'max'],
        'CREDIT_DAY_OVERDUE': ['max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    }

    bureau_agg = bureau_df.groupby(id_col).agg(aggs)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col) for col in bureau_agg.columns]

    return df.merge(bureau_agg, on=id_col, how='left')
```

### 5. Stability-Weighted Ensemble

```python
def stability_weighted_ensemble(models, weights, X_test, historical_ginis):
    """Weight models by their historical stability"""
    # Adjust weights by inverse of variance
    stabilities = [1 / (np.var(ginis) + 0.01) for ginis in historical_ginis]
    adjusted_weights = [w * s for w, s in zip(weights, stabilities)]
    adjusted_weights = np.array(adjusted_weights) / sum(adjusted_weights)

    predictions = np.zeros(len(X_test))
    for model, weight in zip(models, adjusted_weights):
        predictions += weight * model.predict_proba(X_test)[:, 1]

    return predictions
```

---

## Technical Insights

### Gini vs AUC

```
Gini = 2 * AUC - 1
```

| AUC | Gini | Interpretation |
|-----|------|----------------|
| 0.5 | 0 | Random |
| 0.75 | 0.5 | Good |
| 0.85 | 0.7 | Very good |
| 1.0 | 1.0 | Perfect |

### Sources of Instability

| Source | Impact |
|--------|--------|
| Concept drift | Target relationship changes |
| Data drift | Feature distributions shift |
| Sample bias | Different populations over time |
| External factors | Economic conditions |

### Production Considerations

This competition mimicked real-world requirements:
- Model must work on future data
- Performance should be predictable
- Regulatory compliance
- Business planning needs stable metrics

### Key Insight

The novel metric fundamentally changed the competition:
- Can't just maximize Gini
- Must balance accuracy vs consistency
- Feature selection becomes crucial
- Simpler models often more stable

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508337) |
| 13th | [Discussion](https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508113) |

---

## Citation

```bibtex
@misc{home-credit-credit-risk-model-stability,
    author = {Home Credit},
    title = {Home Credit - Credit Risk Model Stability},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/home-credit-credit-risk-model-stability}},
    note = {Kaggle}
}
```
