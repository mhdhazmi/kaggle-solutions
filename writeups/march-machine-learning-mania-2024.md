# March Machine Learning Mania 2024

> Forecast the 2024 College Basketball Tournaments

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Probabilistic Forecasting / Bracket Prediction |
| **Data Domain** | Sports Analytics / Basketball |
| **ML Approach** | Gradient Boosting + Statistical Models |
| **Key Techniques** | Elo Ratings, Portfolio Optimization, Ensemble Methods |
| **Difficulty** | Intermediate-Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 820 |
| **Timeline** | February - April 2024 |
| **Evaluation Metric** | Average Brier Bracket Score |
| **Host** | Kaggle |

## Problem Description

Predict outcomes of both the men's and women's 2024 NCAA basketball tournaments by submitting a portfolio of up to 100,000 brackets.

### The Challenge

- Forecast tournament outcomes for both men's and women's brackets
- Submit portfolio of 1-100,000 brackets
- Optimize for Brier score across all rounds
- Handle tournament structure constraints

### New Format for 2024

Unlike previous years with probability predictions:
- Submit actual brackets (winner predictions)
- Portfolio approach allows hedging
- Implied probabilities calculated from bracket frequencies

---

## Data Description

### Historical Data Provided

| File Prefix | Description |
|-------------|-------------|
| `M*` | Men's tournament data |
| `W*` | Women's tournament data |
| No prefix | Shared data (Cities, Conferences) |

### Key Files

| File | Description |
|------|-------------|
| `*RegularSeasonCompactResults` | Game-by-game results |
| `*RegularSeasonDetailedResults` | Box score statistics |
| `*NCAATourneyCompactResults` | Historical tournament results |
| `*Teams` | Team information |
| `*Seasons` | Season metadata |
| `*MasseyOrdinals` | Third-party rankings |
| `*TeamSpellings` | Name mappings for external data |

### Tournament Structure

- 68 teams (4 play-in games)
- Regions: W, X, Y, Z
- Seeds: 01-16 per region
- 6 rounds: R1, R2, R3 (Sweet 16), R4 (Elite 8), R5 (Final Four), R6 (Championship)

---

## Evaluation

**Metric**: Average Brier Bracket Score

```python
def brier_bracket_score(brackets, ground_truth, rounds=6):
    """
    Score portfolio of brackets using Brier score per round.

    brackets: List of bracket predictions
    ground_truth: Actual tournament results
    """
    round_scores = []

    for round_num in range(1, rounds + 1):
        # Get implied probabilities from bracket portfolio
        implied_probs = compute_implied_probabilities(brackets, round_num)

        # Compute Brier score for this round
        brier = 0
        for team in teams_in_round:
            p = implied_probs.get(team, 0)  # Probability team wins round
            y = 1 if team in ground_truth[round_num] else 0
            brier += (p - y) ** 2

        brier /= num_teams_in_round
        round_scores.append(brier)

    return np.mean(round_scores)

def compute_implied_probabilities(brackets, round_num):
    """Probability = fraction of brackets with team as winner"""
    counts = defaultdict(int)
    for bracket in brackets:
        winners = get_round_winners(bracket, round_num)
        for team in winners:
            counts[team] += 1

    total = len(brackets)
    return {team: count / total for team, count in counts.items()}
```

Lower Brier score is better (unlike most Kaggle competitions).

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $25,000 |
| 2nd | $15,000 |
| 3rd | $10,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Elo rating system with home court adjustments
- Ken Pomeroy (KenPom) style efficiency metrics
- Large portfolio with strategic bracket diversity
- Upset probability calibration

### 2nd Place Solution

**Technique**:
- Historical upset frequency by seed matchup
- Team strength features from multiple sources
- Monte Carlo simulation for bracket generation

### 3rd Place Solution

**Innovation**:
- Massey ordinals aggregation
- Bayesian inference for win probabilities
- Portfolio optimization for Brier score

---

## Common Winning Strategies

### 1. Elo Rating System

```python
class EloRating:
    def __init__(self, k_factor=32, home_advantage=100):
        self.ratings = defaultdict(lambda: 1500)
        self.k_factor = k_factor
        self.home_advantage = home_advantage

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, team_a, team_b, score_a, score_b, neutral=True):
        # Margin of victory adjustment
        mov = abs(score_a - score_b)
        mov_mult = np.log(mov + 1)

        rating_a = self.ratings[team_a]
        rating_b = self.ratings[team_b]

        expected_a = self.expected_score(rating_a, rating_b)
        actual_a = 1 if score_a > score_b else 0

        delta = self.k_factor * mov_mult * (actual_a - expected_a)
        self.ratings[team_a] += delta
        self.ratings[team_b] -= delta

    def predict_game(self, team_a, team_b):
        return self.expected_score(
            self.ratings[team_a],
            self.ratings[team_b]
        )
```

### 2. KenPom-Style Efficiency Metrics

```python
def calculate_efficiency_metrics(games_df, team_id):
    """Calculate offensive/defensive efficiency"""
    team_games = games_df[
        (games_df['WTeamID'] == team_id) |
        (games_df['LTeamID'] == team_id)
    ]

    off_eff = []
    def_eff = []

    for _, game in team_games.iterrows():
        if game['WTeamID'] == team_id:
            pts_for = game['WScore']
            pts_against = game['LScore']
        else:
            pts_for = game['LScore']
            pts_against = game['WScore']

        # Estimate possessions
        poss = estimate_possessions(game)

        # Points per 100 possessions
        off_eff.append(100 * pts_for / poss)
        def_eff.append(100 * pts_against / poss)

    return {
        'adj_o': np.mean(off_eff),
        'adj_d': np.mean(def_eff),
        'net_rating': np.mean(off_eff) - np.mean(def_eff)
    }
```

### 3. Portfolio Bracket Generation

```python
def generate_bracket_portfolio(win_probs, n_brackets=10000):
    """Generate diverse portfolio of brackets"""
    brackets = []

    for _ in range(n_brackets):
        bracket = {}

        for round_num in range(1, 7):
            matchups = get_matchups(round_num, bracket)

            for game_id, (team_a, team_b) in matchups.items():
                prob_a = win_probs.get((team_a, team_b), 0.5)

                # Sample winner based on probability
                if np.random.random() < prob_a:
                    bracket[game_id] = team_a
                else:
                    bracket[game_id] = team_b

        brackets.append(bracket)

    return brackets

def optimize_portfolio(brackets, target_probs):
    """Adjust portfolio to match target probabilities"""
    # Add/remove brackets to shift implied probabilities
    # toward optimal Brier score minimization
    pass
```

### 4. Historical Upset Analysis

```python
def upset_probability_by_seed(historical_games):
    """Calculate upset frequency by seed matchup"""
    upset_rates = {}

    for seed_high, seed_low in itertools.combinations(range(1, 17), 2):
        matchup_games = historical_games[
            (historical_games['WSeed'] == seed_high) &
            (historical_games['LSeed'] == seed_low) |
            (historical_games['WSeed'] == seed_low) &
            (historical_games['LSeed'] == seed_high)
        ]

        if len(matchup_games) > 0:
            lower_seed_wins = (matchup_games['WSeed'] == seed_low).sum()
            upset_rates[(seed_high, seed_low)] = lower_seed_wins / len(matchup_games)

    return upset_rates

# Famous upset probabilities (historical)
ROUND_1_UPSETS = {
    (1, 16): 0.01,   # Very rare (UMBC!)
    (2, 15): 0.06,
    (3, 14): 0.15,
    (4, 13): 0.21,
    (5, 12): 0.35,   # Famous upset seed
    (6, 11): 0.37,
    (7, 10): 0.39,
    (8, 9): 0.49,    # Nearly coin flip
}
```

### 5. Feature Engineering

```python
def create_tournament_features(team_id, season):
    """Features for tournament prediction"""
    features = {}

    # Regular season record
    features['win_pct'] = calculate_win_pct(team_id, season)

    # Strength of schedule
    features['sos'] = calculate_sos(team_id, season)

    # Recent form (last 10 games)
    features['recent_form'] = calculate_recent_form(team_id, season)

    # Efficiency metrics
    eff = calculate_efficiency_metrics(games, team_id)
    features.update(eff)

    # Tournament experience
    features['tourney_appearances'] = count_recent_tourneys(team_id)

    # Coaching
    features['coach_tourney_wins'] = get_coach_tourney_wins(team_id)

    # Rankings
    features['ap_rank'] = get_ap_rank(team_id, season)
    features['kenpom_rank'] = get_kenpom_rank(team_id, season)

    return features
```

---

## Technical Insights

### Seed vs Seed Historical Win Rates

| Matchup | Higher Seed Win % |
|---------|-------------------|
| 1 vs 16 | 99% |
| 2 vs 15 | 94% |
| 3 vs 14 | 85% |
| 4 vs 13 | 79% |
| 5 vs 12 | 65% |
| 6 vs 11 | 63% |
| 7 vs 10 | 61% |
| 8 vs 9 | 51% |

### Key Predictive Features

| Feature | Importance |
|---------|------------|
| KenPom adjusted efficiency | Very High |
| Seed | High |
| Strength of schedule | High |
| Tournament experience | Medium |
| Recent form | Medium |

### Portfolio Strategy

| Approach | Pro | Con |
|----------|-----|-----|
| All chalk | Safe, low variance | Misses upsets |
| Heavy upsets | Big wins if right | Big losses if wrong |
| Diverse portfolio | Balanced | Complex to optimize |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | Not specified |
| Internet | Disabled |
| GPU | Available |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/493879) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/493893) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/493839) |

---

## Citation

```bibtex
@misc{march-machine-learning-mania-2024,
    author = {Kaggle},
    title = {March Machine Learning Mania 2024},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/march-machine-learning-mania-2024}},
    note = {Kaggle}
}
```
