# Child Mind Institute — Problematic Internet Use

> Relating Physical Activity to Problematic Internet Use

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Ordinal Classification / Regression |
| **Data Domain** | Healthcare / Behavioral Science |
| **ML Approach** | Gradient Boosting + Neural Networks |
| **Key Techniques** | Time Series Features, Threshold Optimization, QWK Optimization |
| **Difficulty** | Intermediate-Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $60,000 |
| **Teams** | 3,559 |
| **Timeline** | 2024 |
| **Evaluation Metric** | Quadratic Weighted Kappa (QWK) |
| **Host** | Child Mind Institute |

## Problem Description

Predict the severity of problematic internet use in children based on physical activity data from wearable devices.

### The Challenge

- Use accelerometer data to predict internet addiction severity
- Handle time series from fitness trackers
- Ordinal classification (severity levels)
- Link physical activity patterns to behavioral outcomes

### Why It Matters

- **Mental Health**: Early identification of problematic behaviors
- **Prevention**: Intervention before severe addiction
- **Research**: Understand activity-behavior relationships
- **Scalability**: Passive monitoring via wearables

---

## Data Description

### Dataset Structure

| Component | Description |
|-----------|-------------|
| Accelerometer data | Time series from wearable devices |
| Demographics | Age, gender, etc. |
| Survey responses | Questionnaire data |
| Target | Severity level (ordinal) |

### Severity Levels

Ordinal target with multiple levels:
- 0: None/minimal
- 1: Mild
- 2: Moderate
- 3: Severe

### Feature Types

| Category | Examples |
|----------|----------|
| Activity metrics | Steps, active minutes, sedentary time |
| Sleep patterns | Duration, quality, timing |
| Demographics | Age, sex |
| Survey scores | Screen time questionnaires |

---

## Evaluation

**Metric**: Quadratic Weighted Kappa (QWK)

QWK measures agreement between predicted and actual ordinal ratings:
- Accounts for ordinal nature of target
- Penalizes predictions far from true label more heavily
- Range: -1 to 1 (1 = perfect agreement)

### QWK Formula

```
κ = 1 - (Σᵢⱼ wᵢⱼ Oᵢⱼ) / (Σᵢⱼ wᵢⱼ Eᵢⱼ)
```

Where weights wᵢⱼ = (i-j)² (quadratic penalty)

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |

**Total Prize Pool**: $60,000

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Time series feature engineering
- Gradient boosting with QWK optimization
- Threshold optimization for ordinal targets

### 2nd Place Solution

**Technique**:
- TabNet for tabular data
- Activity pattern embeddings
- Ensemble with diverse models

### 3rd Place Solution

**Innovation**:
- CNN on raw accelerometer data
- Multi-task learning
- Pseudo-labeling

---

## Common Winning Strategies

### 1. Time Series Feature Engineering

```python
# Activity statistics
df['mean_steps'] = accel_data.groupby('id')['steps'].mean()
df['std_steps'] = accel_data.groupby('id')['steps'].std()
df['max_steps'] = accel_data.groupby('id')['steps'].max()

# Temporal patterns
df['morning_activity'] = morning_accel.mean()
df['evening_activity'] = evening_accel.mean()
df['activity_variance'] = daily_steps.var()
```

### 2. QWK Optimization

Direct optimization or threshold tuning:
```python
from scipy.optimize import minimize

def qwk_loss(thresholds, y_true, y_pred_proba):
    y_pred = apply_thresholds(y_pred_proba, thresholds)
    return -quadratic_weighted_kappa(y_true, y_pred)

optimal_thresholds = minimize(qwk_loss, initial_thresholds,
                               args=(y_val, y_pred_proba))
```

### 3. Ordinal Classification Approaches

| Approach | Description |
|----------|-------------|
| OrdinalEncoder + Regression | Treat as regression, threshold |
| Multiple binary classifiers | P(y≥k) for each k |
| Direct QWK optimization | Custom loss function |
| Neural network ordinal | Ordinal output layer |

### 4. Handling Missing Data

Accelerometer data often has gaps:
- Imputation with rolling statistics
- Flag missing periods as features
- Use robust aggregations

### 5. Activity Pattern Features

```python
# Regularity
df['activity_entropy'] = calc_entropy(daily_patterns)

# Sleep indicators
df['sleep_duration'] = estimate_sleep(accel_night)
df['sleep_regularity'] = sleep_times.std()

# Sedentary patterns
df['sedentary_bouts'] = count_sedentary_periods(accel)
```

---

## Technical Insights

### Accelerometer Data Processing

Raw accelerometer provides x, y, z acceleration:
```python
# Compute magnitude
magnitude = np.sqrt(x**2 + y**2 + z**2)

# Activity classification
activity_counts = count_above_threshold(magnitude, threshold)
```

### QWK vs Accuracy

| Metric | Behavior |
|--------|----------|
| Accuracy | All errors equal |
| QWK | Distant errors penalized more |
| Kappa | Accounts for chance agreement |

### Ordinal Target Handling

For ordinal targets like severity levels:
- Don't use standard multi-class
- Preserve ordering information
- Threshold regression predictions

### Domain Knowledge

Physical activity research suggests:
- Regular activity associated with better mental health
- Sleep patterns predict behavioral issues
- Sedentary time linked to screen time
- Circadian rhythm disruption matters

---

## Code Requirements

| Requirement | Limit |
|-------------|-------|
| CPU/GPU Runtime | Standard limits |
| Internet | Disabled |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552638) |
| 2nd | [Discussion](https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552712) |
| 3rd | [Discussion](https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552677) |
| 4th | [Discussion](https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552861) |
| 5th | [Discussion](https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552656) |

---

## Citation

```bibtex
@misc{child-mind-institute-problematic-internet-use,
    title = {Child Mind Institute — Problematic Internet Use},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/child-mind-institute-problematic-internet-use}},
    note = {Child Mind Institute, Kaggle}
}
```
