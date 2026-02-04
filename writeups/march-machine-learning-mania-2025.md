# March Machine Learning Mania 2025

> Forecast the 2025 NCAA Basketball Tournaments

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Probabilistic Prediction / Sports Forecasting |
| **Data Domain** | Sports Analytics / Basketball |
| **ML Approach** | Gradient Boosting + Elo Ratings |
| **Key Techniques** | Historical Analysis, Probability Calibration, Ensemble |
| **Difficulty** | Intermediate |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Prediction Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,727 |
| **Timeline** | February - April 2025 |
| **Evaluation Metric** | Brier Score (Mean Squared Error) |
| **Host** | Kaggle |

## Problem Description

Forecast the outcomes of both the men's and women's 2025 NCAA basketball tournaments by predicting win probabilities for every possible matchup.

### The Challenge

- Predict probability that Team A beats Team B
- Cover all possible tournament matchups
- Both men's and women's tournaments
- Combine historical data with current season performance

### March Madness Appeal

- **Unpredictability**: Upsets are common ("Cinderella stories")
- **Single elimination**: One bad game ends your run
- **National attention**: Huge cultural event in the US

---

## Data Description

### Dataset Structure

Comprehensive historical data provided:

| Category | Files |
|----------|-------|
| Team Info | MTeams.csv, WTeams.csv |
| Seeds | MNCAATourneySeeds.csv, WNCAATourneySeeds.csv |
| Game Results | MRegularSeasonDetailedResults.csv, etc. |
| Conferences | MConferences.csv, WConferences.csv |
| Rankings | MMasseyOrdinals.csv (various ranking systems) |

### Key Data Files

| File | Description |
|------|-------------|
| `Teams.csv` | Team IDs and names |
| `Seeds.csv` | Tournament seedings (1-16) |
| `RegularSeasonResults.csv` | Regular season game outcomes |
| `TourneyResults.csv` | Historical tournament results |
| `MasseyOrdinals.csv` | Various ranking systems |

### ID Convention

- Men's teams: 1000-1999
- Women's teams: 3000-3999

### Submission Format

```csv
ID,Pred
2025_1101_1102,0.65
2025_1101_1103,0.72
...
```

Where:
- ID = `{Season}_{LowerTeamID}_{HigherTeamID}`
- Pred = P(Lower Team ID wins)

---

## Evaluation

**Metric**: Brier Score (equivalent to MSE for probabilities)

```
Brier Score = (1/N) × Σ(pᵢ - oᵢ)²
```

Where:
- pᵢ = predicted probability
- oᵢ = outcome (0 or 1)

### Brier Score Properties

- Range: 0 (perfect) to 1 (worst)
- Rewards well-calibrated probabilities
- Penalizes confident wrong predictions heavily

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $10,000 |
| 2nd | $8,000 |
| 3rd | $7,000 |
| 4th-8th | $5,000 each |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Elo rating system foundation
- Gradient boosting for adjustments
- Team strength relative to seed expectation

### 2nd Place Solution

**Technique**:
- Multiple ranking system ensemble
- Bayesian probability calibration
- Recent form weighting

### 3rd Place Solution

**Innovation**:
- Neural network for matchup prediction
- Feature engineering on historical upsets
- Conference strength adjustments

---

## Common Winning Strategies

### 1. Elo Rating System

Classic approach for head-to-head predictions:
```python
def elo_expected(rating_a, rating_b):
    return 1 / (1 + 10**((rating_b - rating_a) / 400))

def elo_update(rating, expected, actual, k=32):
    return rating + k * (actual - expected)
```

### 2. Feature Engineering

| Feature | Description |
|---------|-------------|
| Seed difference | |seed_A - seed_B| |
| Elo ratings | Team strength ratings |
| Win percentage | Season win rate |
| Strength of schedule | Quality of opponents |
| Offensive/Defensive efficiency | Points per 100 possessions |
| Tempo | Pace of play |

### 3. External Rankings

Massey Ordinals includes many ranking systems:
- Sagarin
- KenPom
- RPI
- BPI
- NET

Ensemble these for robust predictions.

### 4. Probability Calibration

Ensure predictions are well-calibrated:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method='isotonic')
```

### 5. Seed-Based Baseline

Historical upset rates by seed matchup:
```
#1 vs #16: 99% favorite wins
#2 vs #15: 94%
#5 vs #12: 65% (famous "12-over-5 upset")
#8 vs #9:  50% (essentially coin flip)
```

---

## Technical Insights

### Tournament Structure

- 64 teams per gender (68 with play-in)
- Single elimination bracket
- Seeded 1-16 in four regions
- Final Four → Championship

### Key Factors

1. **Seed**: Most predictive single feature
2. **Recent form**: Last 10 games matter
3. **Experience**: Tournament experience helps
4. **Location**: Travel distance affects performance
5. **Matchup**: Style of play interactions

### Upset Prediction

Characteristics of upset-prone favorites:
- Inconsistent season performance
- Poor free throw shooting
- Weak perimeter defense
- Inexperienced roster

### Men's vs Women's

| Aspect | Men's | Women's |
|--------|-------|---------|
| Parity | More upsets | Top teams dominate more |
| Data availability | More historical data | Growing dataset |
| Prediction difficulty | Harder | Easier (more predictable) |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572717) |
| 2nd | [Discussion](https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572528) |
| 3rd | [Discussion](https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572553) |
| 4th | [Discussion](https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572466) |
| 5th | [Discussion](https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572909) |

---

## Citation

```bibtex
@misc{march-machine-learning-mania-2025,
    author = {Jeff Sonas, Paul Mooney, Addison Howard, and Will Cukierski},
    title = {March Machine Learning Mania 2025},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/march-machine-learning-mania-2025}},
    note = {Kaggle}
}
```
