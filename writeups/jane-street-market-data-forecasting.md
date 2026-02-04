# Jane Street Real-Time Market Data Forecasting

> Predict financial market responders using real-world data

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression / Time Series Forecasting |
| **Data Domain** | Finance / Quantitative Trading |
| **ML Approach** | Neural Networks + Gradient Boosting |
| **Key Techniques** | Feature Engineering, Online Learning, Ensemble |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $120,000 |
| **Teams** | 3,757 |
| **Timeline** | 2025 |
| **Evaluation Metric** | Weighted Zero-Mean R-squared Score |
| **Host** | Jane Street |

## Problem Description

Predict financial market "responders" - target variables that represent market movements - using real-time market data. This competition simulates the challenges of quantitative trading.

### The Challenge

- Process streaming market data in real-time
- Predict multiple responder variables
- Handle noisy, non-stationary financial data
- Optimize for weighted R-squared metric

### Why It Matters

- Advance quantitative finance research
- Develop robust forecasting methods
- Handle real-world trading constraints

---

## Data Description

### Dataset Structure

| Feature Type | Description |
|--------------|-------------|
| Market features | Real anonymized market data |
| Responders | Target variables to predict |
| Timestamps | Time-ordered observations |
| Weights | Sample importance weights |

### Key Characteristics

- **Anonymized**: Feature names obscured
- **Real data**: Actual market observations
- **Time-ordered**: Must respect temporal ordering
- **Weighted evaluation**: Not all samples equally important

---

## Evaluation

**Metric**: Weighted Zero-Mean R-squared Score

```
R² = 1 - Σ(w × (y - ŷ)²) / Σ(w × y²)
```

Where:
- `w` = sample weights
- `y` = actual values
- `ŷ` = predicted values

**Zero-mean**: Predictions centered around zero expected return.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $30,000 |
| 2nd | $25,000 |
| 3rd | $20,000 |
| 4th | $15,000 |
| 5th | $10,000 |
| 6th-10th | $4,000 each |

---

## Top Solutions

### 8th Place: Evgeniia Grigoreva

**Key Approach**:
- Feature engineering on anonymized data
- Neural network architecture for temporal patterns
- Ensemble of multiple model types
- Careful handling of sample weights

*(Full details in linked writeup)*

---

## Common Winning Strategies

### 1. Feature Engineering
Creating meaningful features from anonymized data through:
- Statistical aggregations
- Lag features
- Rolling statistics
- Cross-feature interactions

### 2. Neural Network Architectures
- LSTM/GRU for temporal dependencies
- Transformer attention mechanisms
- MLP with careful regularization

### 3. Gradient Boosting
- LightGBM for tabular patterns
- XGBoost for robustness
- CatBoost for categorical handling

### 4. Online Learning
Adapting models to recent data distribution shifts.

### 5. Ensemble Methods
Combining diverse models for robust predictions.

---

## Technical Insights

### Challenges in Financial Forecasting

1. **Non-stationarity**: Market dynamics change over time
2. **Low signal-to-noise**: Most "patterns" are noise
3. **Regime changes**: Market behavior shifts dramatically
4. **Overfitting risk**: Easy to fit to historical patterns that don't persist

### Weighted R² Considerations

- Focus on high-weight samples
- Some time periods matter more than others
- Calibrate predictions to expected returns

---

## Technical Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | ≤9 hours |
| GPU Runtime | ≤9 hours |
| Internet Access | Disabled |
| Real-time API | Must process streaming data |

---

## Solution Links

| Place | Author | Solution |
|-------|--------|----------|
| 8th | Evgeniia Grigoreva | [Writeup](https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution) |

---

## Citation

```bibtex
@misc{jane-street-real-time-market-data-forecasting,
    title = {Jane Street Real-Time Market Data Forecasting},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/jane-street-real-time-market-data-forecasting}},
    note = {Kaggle, Jane Street}
}
```
