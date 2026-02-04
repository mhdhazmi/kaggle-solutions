# Optiver - Trading at the Close

> Predict US stocks closing movements

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Time Series Regression |
| **Data Domain** | Finance / Stock Market |
| **ML Approach** | Gradient Boosting + Neural Networks |
| **Key Techniques** | Order Book Features, Temporal Features, Cross-Sectional Features |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $100,000 |
| **Teams** | 4,436 |
| **Timeline** | September 2023 - March 2024 |
| **Evaluation Metric** | MAE |
| **Host** | Optiver |

## Problem Description

Predict the closing price movements for hundreds of Nasdaq-listed stocks using order book and closing auction data from the final 10 minutes of trading.

### The Challenge

- Predict 60-second future price movements
- Use order book data (bids, asks, imbalances)
- Handle the critical market close period
- Work with high-frequency financial data

### Why It Matters

- **Market Making**: Better predictions improve liquidity provision
- **Price Discovery**: Aid in fair closing price determination
- **Trading Strategy**: Optimize execution during volatile close
- **Market Stability**: Reduce volatility spikes

---

## Data Description

### The Nasdaq Closing Cross

The daily closing auction process that establishes official closing prices. Crucial for:
- Index calculations
- Mutual fund NAV
- Portfolio valuations
- Derivative settlements

### Data Fields

| Column | Description |
|--------|-------------|
| `stock_id` | Stock identifier |
| `date_id` | Date identifier |
| `seconds_in_bucket` | Seconds since market close starts |
| `imbalance_size` | Unmatched shares in auction |
| `imbalance_buy_sell_flag` | Direction of imbalance |
| `reference_price` | Auction reference price |
| `matched_size` | Matched shares in auction |
| `far_price` | Crossing price (far from market) |
| `near_price` | Crossing price (near market) |
| `bid_price` / `ask_price` | Best bid/ask |
| `bid_size` / `ask_size` | Size at best bid/ask |
| `wap` | Weighted average price |
| `target` | 60-second future price return (basis points) |

### Time Structure

- Data from final 10 minutes of trading (T-10 to T-0)
- Predictions made at 10-second intervals
- Target is price movement over next 60 seconds

---

## Evaluation

**Metric**: Mean Absolute Error (MAE)

```python
import numpy as np

def mae(y_true, y_pred):
    """Mean Absolute Error in basis points"""
    return np.mean(np.abs(y_true - y_pred))
```

Lower is better. Target values are in basis points.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $30,000 |
| 2nd | $20,000 |
| 3rd | $15,000 |
| 4th | $12,000 |
| 5th | $10,000 |
| 6th-10th | $2,600 each |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- LightGBM + Neural Network ensemble
- Extensive order book features
- Cross-stock information
- Time-aware feature engineering

### 2nd Place Solution

**Technique**:
- TabNet deep learning model
- Imbalance dynamics features
- Stock grouping by sector

### 3rd Place Solution

**Innovation**:
- Transformer for sequence modeling
- Order flow imbalance signals
- Market microstructure features

---

## Common Winning Strategies

### 1. Order Book Feature Engineering

```python
def create_orderbook_features(df):
    """Features from order book data"""
    features = {}

    # Spread features
    features['spread'] = df['ask_price'] - df['bid_price']
    features['spread_pct'] = features['spread'] / df['wap']
    features['mid_price'] = (df['bid_price'] + df['ask_price']) / 2

    # Size features
    features['total_size'] = df['bid_size'] + df['ask_size']
    features['size_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1)

    # Price pressure
    features['bid_pressure'] = df['bid_size'] * df['bid_price']
    features['ask_pressure'] = df['ask_size'] * df['ask_price']
    features['pressure_imbalance'] = (features['bid_pressure'] - features['ask_pressure']) / \
                                     (features['bid_pressure'] + features['ask_pressure'] + 1)

    # WAP deviation
    features['wap_bid_diff'] = df['wap'] - df['bid_price']
    features['wap_ask_diff'] = df['ask_price'] - df['wap']

    return pd.DataFrame(features)
```

### 2. Auction Imbalance Features

```python
def create_imbalance_features(df):
    """Features from closing auction imbalances"""
    features = {}

    # Imbalance ratios
    features['imbalance_ratio'] = df['imbalance_size'] / (df['matched_size'] + 1)
    features['imbalance_direction'] = df['imbalance_buy_sell_flag']

    # Price relationships
    features['ref_to_wap'] = (df['reference_price'] - df['wap']) / df['wap']
    features['far_near_spread'] = df['far_price'] - df['near_price']

    # Auction pressure
    features['auction_pressure'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']

    # Matched to total ratio
    features['match_ratio'] = df['matched_size'] / (df['matched_size'] + df['imbalance_size'] + 1)

    return pd.DataFrame(features)
```

### 3. Temporal Features

```python
def create_temporal_features(df):
    """Time-based features for closing period"""
    features = {}

    # Time to close
    features['seconds_to_close'] = 600 - df['seconds_in_bucket']
    features['minutes_to_close'] = features['seconds_to_close'] / 60

    # Cyclical encoding
    features['time_sin'] = np.sin(2 * np.pi * df['seconds_in_bucket'] / 600)
    features['time_cos'] = np.cos(2 * np.pi * df['seconds_in_bucket'] / 600)

    # Time buckets
    features['time_bucket'] = df['seconds_in_bucket'] // 60

    # Urgency (inverse time)
    features['urgency'] = 1 / (features['seconds_to_close'] + 1)

    return pd.DataFrame(features)

def create_rolling_features(df, windows=[3, 6, 12]):
    """Rolling statistics over time buckets"""
    features = {}

    for window in windows:
        # Rolling means
        features[f'wap_ma_{window}'] = df.groupby('stock_id')['wap'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Rolling volatility
        features[f'wap_std_{window}'] = df.groupby('stock_id')['wap'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

        # Momentum
        features[f'wap_momentum_{window}'] = df.groupby('stock_id')['wap'].transform(
            lambda x: x.pct_change(window)
        )

    return pd.DataFrame(features)
```

### 4. Cross-Sectional Features

```python
def create_cross_sectional_features(df):
    """Features comparing stock to market"""
    features = {}

    # Market-wide aggregates per time bucket
    market_stats = df.groupby(['date_id', 'seconds_in_bucket']).agg({
        'wap': 'mean',
        'imbalance_size': 'mean',
        'spread': 'mean',
        'target': 'mean'
    }).reset_index()

    market_stats.columns = ['date_id', 'seconds_in_bucket',
                            'market_wap', 'market_imbalance', 'market_spread', 'market_target']

    # Merge back
    df = df.merge(market_stats, on=['date_id', 'seconds_in_bucket'])

    # Relative features
    features['wap_vs_market'] = df['wap'] / df['market_wap']
    features['imbalance_vs_market'] = df['imbalance_size'] / (df['market_imbalance'] + 1)
    features['spread_vs_market'] = df['spread'] / (df['market_spread'] + 1e-8)

    return pd.DataFrame(features)
```

### 5. LightGBM Model

```python
from lightgbm import LGBMRegressor

def train_lgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM for price prediction"""

    params = {
        'objective': 'mae',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 127,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }

    model = LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    return model
```

### 6. Neural Network Model

```python
import torch
import torch.nn as nn

class StockPredictionNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

# Training with MAE loss
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

---

## Technical Insights

### Market Microstructure

| Concept | Description |
|---------|-------------|
| Order Imbalance | Difference between buy/sell orders |
| Price Impact | How orders move prices |
| Liquidity | Ease of executing trades |
| Volatility | Price fluctuation magnitude |

### Closing Auction Dynamics

| Time | Activity |
|------|----------|
| T-10 min | Imbalance info published |
| T-5 min | Orders can be modified |
| T-0 | Auction executes |

### Feature Importance (Typical)

| Feature Type | Importance |
|--------------|------------|
| Imbalance features | Very High |
| Time features | High |
| Spread features | Medium-High |
| Size features | Medium |
| Cross-sectional | Medium |

### Challenges

| Challenge | Solution |
|-----------|----------|
| Non-stationarity | Rolling features, recent data emphasis |
| Low signal-to-noise | Ensemble methods |
| Regime changes | Robust features |
| Overfitting | Strong regularization |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | 9 hours |
| GPU Runtime | 9 hours |
| Internet | Disabled |
| Time-series API | Must use provided API |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/488703) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/488891) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/488618) |
| 4th | [Discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/489170) |
| 5th | [Discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/489119) |

---

## Citation

```bibtex
@misc{optiver-trading-at-the-close,
    author = {Optiver},
    title = {Optiver - Trading at the Close},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/optiver-trading-at-the-close}},
    note = {Kaggle}
}
```
