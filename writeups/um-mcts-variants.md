# UM - Game-Playing Strength of MCTS Variants

> Predict which variants of Monte-Carlo Tree Search will perform well or poorly against each other in hundreds of board games

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression / Meta-Learning |
| **Data Domain** | Game AI / Algorithm Performance |
| **ML Approach** | Gradient Boosting + Neural Networks |
| **Key Techniques** | Feature Engineering on Game Properties, Meta-Learning |
| **Difficulty** | Intermediate-Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,608 |
| **Timeline** | 2024 |
| **Evaluation Metric** | Mean Squared Error (MSE) |
| **Host** | University of Maastricht |

## Problem Description

Predict the relative game-playing strength of different Monte-Carlo Tree Search (MCTS) algorithm variants across hundreds of different board games.

### The Challenge

- Predict win rates between MCTS variants
- Generalize across diverse board games
- Understand algorithm-game interactions
- Meta-learning across games

### Why It Matters

- **AI Research**: Understand algorithm design choices
- **Game AI**: Automatic algorithm selection
- **Meta-Learning**: Learning to learn
- **Computational Efficiency**: Avoid expensive evaluations

---

## Data Description

### Dataset Structure

| Component | Description |
|-----------|-------------|
| Games | Hundreds of different board games |
| MCTS variants | Different algorithm configurations |
| Match results | Win rates from actual game play |
| Game features | Properties of each game |

### MCTS Variants

Different configurations of Monte-Carlo Tree Search:
- Selection policies (UCB1, PUCT, etc.)
- Expansion strategies
- Simulation policies
- Backpropagation methods

### Game Features

| Category | Examples |
|----------|----------|
| Structure | Board size, branching factor |
| Rules | Capture mechanics, win conditions |
| Complexity | State space, decision complexity |
| Type | Abstract strategy, area control, etc. |

---

## Evaluation

**Metric**: Mean Squared Error (MSE)

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

Predicting continuous win rates (0 to 1).

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | TBD |
| 2nd | TBD |
| 3rd | TBD |

**Total Prize Pool**: $50,000

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Extensive feature engineering on games
- Meta-features about MCTS variants
- Gradient boosting ensemble

### 2nd Place Solution

**Technique**:
- Neural network for game embeddings
- Interaction features between game and algorithm
- Careful cross-validation

### 3rd Place Solution

**Innovation**:
- Game similarity measures
- Transfer learning between similar games
- Ensemble with diverse models

---

## Common Winning Strategies

### 1. Game Feature Engineering

```python
# Game complexity features
df['state_space_size'] = game.estimated_states
df['branching_factor'] = game.avg_legal_moves
df['game_length'] = game.avg_turns

# Structural features
df['board_size'] = game.board.size
df['num_piece_types'] = len(game.piece_types)
df['has_captures'] = game.allows_captures
```

### 2. Algorithm Features

```python
# MCTS variant properties
df['selection_policy'] = variant.selection
df['exploration_constant'] = variant.exploration_c
df['use_progressive_bias'] = variant.progressive_bias
df['playout_policy'] = variant.simulation_policy
```

### 3. Interaction Features

```python
# Game-Algorithm interactions
df['complexity_x_exploration'] = df['state_space'] * df['exploration_c']
df['branching_x_selection'] = df['branching_factor'] * encode(df['selection'])
```

### 4. Meta-Learning Approach

| Level | Features |
|-------|----------|
| Game | Properties of the game |
| Algorithm | MCTS variant configuration |
| Matchup | Algorithm A vs Algorithm B |
| Prediction | Expected win rate |

### 5. Cross-Validation Strategy

Important to handle game clustering:
- Leave-game-out CV
- Stratified by game type
- Avoid leakage between similar games

---

## Technical Insights

### Understanding MCTS

Monte-Carlo Tree Search components:
1. **Selection**: Choose promising nodes (UCB1)
2. **Expansion**: Add new nodes to tree
3. **Simulation**: Random playouts
4. **Backpropagation**: Update statistics

### Algorithm Comparison

Different variants excel at different game types:
- High branching factor → Careful selection
- Long games → Efficient simulation
- Tactical games → Deep search
- Strategic games → Position evaluation

### Game Taxonomy

Games categorized by:
- **Branching factor**: Low/medium/high
- **Depth**: Short/medium/long
- **Type**: Abstract, territorial, racing
- **Information**: Perfect/imperfect

### Prediction Challenges

- **Novel games**: Unseen game types
- **Algorithm interactions**: Complex relationships
- **Sample efficiency**: Limited match data
- **Transitivity violations**: A beats B, B beats C, C beats A

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549801) |
| 2nd | [Discussion](https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549718) |
| 3rd | [Discussion](https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549588) |
| 4th | [Discussion](https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549603) |
| 5th | [Discussion](https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549585) |

---

## Citation

```bibtex
@misc{um-game-playing-strength-of-mcts-variants,
    title = {UM - Game-Playing Strength of MCTS Variants},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/um-game-playing-strength-of-mcts-variants}},
    note = {University of Maastricht, Kaggle}
}
```
