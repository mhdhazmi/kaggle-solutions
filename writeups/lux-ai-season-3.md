# NeurIPS 2024 - Lux AI Season 3

> Deep space exploration!

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Agent Reinforcement Learning / Game AI |
| **Data Domain** | Simulation / Strategy Game |
| **ML Approach** | RL (PPO, DQN) + Rule-Based Heuristics |
| **Key Techniques** | Self-Play, Imitation Learning, Search |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Simulation Competition |
| **Total Prize** | $50,000 |
| **Teams** | 701 |
| **Timeline** | 2024-2025 |
| **Evaluation Metric** | Elo Rating (1v1 matches) |
| **Host** | Lux AI Challenge |

## Problem Description

Create AI bots to play a novel multi-agent 1v1 strategy game with deep space exploration theme.

### The Game

A turn-based strategy game where players:
- Control multiple units
- Gather resources
- Build structures
- Expand territory
- Defeat opponent

### The Challenge

- Design an AI agent that plays the game
- Compete against other submitted agents
- Adapt to various opponent strategies
- Balance economy, exploration, and combat

### Why It Matters

- **AI Research**: Multi-agent RL benchmark
- **Game AI**: Real-world game development applications
- **Strategy Learning**: Complex decision-making under uncertainty

---

## Game Mechanics

### Core Elements

| Element | Description |
|---------|-------------|
| Units | Mobile agents with various capabilities |
| Resources | Collectible materials for building |
| Buildings | Structures providing advantages |
| Map | Procedurally generated terrain |
| Fog of War | Limited visibility |

### Turn Structure

1. **Observation**: Receive game state
2. **Decision**: Choose actions for all units
3. **Execution**: Actions resolve simultaneously
4. **Update**: Game state advances

### Victory Conditions

- Eliminate opponent's units
- Control majority of map
- Resource/score superiority at turn limit

---

## Evaluation

**Metric**: Elo Rating System

Agents compete in 1v1 matches:
- Win increases Elo
- Loss decreases Elo
- Rating reflects relative skill

### Match System

- Round-robin style tournaments
- Multiple matches per agent pair
- Final ranking by Elo

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
- Deep RL with PPO algorithm
- Extensive self-play training
- Carefully designed reward shaping
- Ensemble of specialized policies

### 2nd Place Solution

**Technique**:
- Imitation learning from rule-based bot
- Fine-tuning with RL
- Action masking for valid moves
- Multi-scale planning

### 3rd Place Solution

**Innovation**:
- Rule-based core with learned components
- Monte Carlo Tree Search
- Opponent modeling

---

## Common Winning Strategies

### 1. Approach Selection

| Approach | Pros | Cons |
|----------|------|------|
| Pure RL | Flexible, learns emergent strategies | Sample inefficient, unstable |
| Rule-based | Reliable, interpretable | Limited adaptation |
| Hybrid | Best of both | Complex to implement |
| Search-based | Optimal play | Computationally expensive |

### 2. Reinforcement Learning Setup

**Algorithm**: PPO (Proximal Policy Optimization)
```python
# Typical setup
obs_space = game.observation_space
act_space = game.action_space

policy = ActorCritic(obs_space, act_space)
agent = PPO(policy, lr=3e-4, n_steps=2048)
```

### 3. Reward Shaping

Critical for learning:
```python
reward = (
    resource_gathered * 0.1 +
    enemy_damage * 1.0 +
    territory_control * 0.5 -
    unit_lost * 2.0 +
    game_win * 100.0
)
```

### 4. Self-Play Training

```
Initial Random Agent
       ↓
Train vs Self → Improved Agent
       ↓
Train vs History of Self → More Robust
       ↓
Final Agent
```

### 5. Action Space Design

Handling large action spaces:
- Hierarchical actions (select unit, then action)
- Autoregressive output
- Action masking for invalid moves

---

## Technical Insights

### Observation Space

What the agent sees:
- Unit positions and health
- Resource locations
- Building states
- Fog of war limitations
- Historical information

### Action Space

What the agent controls:
- Unit movement
- Resource gathering
- Building construction
- Combat targeting
- Strategic positioning

### Key RL Challenges

1. **Large state space**: Complex game states
2. **Sparse rewards**: Win/loss only at end
3. **Multi-agent dynamics**: Opponent adapts
4. **Partial observability**: Fog of war
5. **Long horizons**: Many turns per game

### Compute Requirements

Successful RL approaches typically need:
- Millions of games for training
- GPU for neural network inference
- Efficient simulation environment

### Rule-Based Components

Even RL solutions often include:
- Early game opening strategies
- Combat micro-management
- Resource collection heuristics

---

## Code Requirements

| Requirement | Limit |
|-------------|-------|
| Per-turn time | ~1 second |
| Memory | Limited |
| Internet | Disabled |
| Language | Python (kaggle_environments) |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/lux-ai-season-3/discussion/569562) |
| 2nd | [Discussion](https://www.kaggle.com/c/lux-ai-season-3/discussion/568621) |
| 3rd | [Discussion](https://www.kaggle.com/c/lux-ai-season-3/discussion/568494) |
| 4th | [Discussion](https://www.kaggle.com/c/lux-ai-season-3/discussion/569928) |
| 5th | [Discussion](https://www.kaggle.com/c/lux-ai-season-3/discussion/571111) |

---

## Resources

- [Lux AI Official Documentation](https://github.com/Lux-AI-Challenge/Lux-Design-S3)
- [kaggle_environments](https://github.com/Kaggle/kaggle-environments)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

---

## Citation

```bibtex
@misc{lux-ai-season-3,
    title = {NeurIPS 2024 - Lux AI Season 3},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/lux-ai-season-3}},
    note = {Lux AI Challenge, Kaggle}
}
```
