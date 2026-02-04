# ARC Prize 2024

> Create an AI capable of solving reasoning tasks it has never seen before

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Abstract Reasoning / Pattern Recognition |
| **Data Domain** | AGI Benchmark / Visual Reasoning |
| **ML Approach** | Program Synthesis + Neural-Symbolic Hybrids |
| **Key Techniques** | DSL-based Search, LLM Augmentation, Ensemble Methods |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $600,000 ($100K base + $500K for >85%) |
| **Teams** | 1,716 |
| **Timeline** | June - November 2024 |
| **Evaluation Metric** | Task Solve Rate |
| **Host** | ARC Prize Foundation |

## Problem Description

Develop AI systems that can solve novel abstract reasoning tasks - tasks the system has never seen before.

### The Challenge

- Solve visual reasoning puzzles
- Generalize to completely new problem types
- Demonstrate genuine abstraction capability
- Beat human benchmark (85%)

### Why ARC Matters

The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) is considered one of the most important benchmarks for measuring progress toward AGI:

- **Humans score ~85%** with minimal effort
- **Best AI systems score ~34%** (pre-competition)
- Tests ability to **learn new skills efficiently**
- Designed to be **resistant to memorization**

---

## Data Description

### Task Format

Each task consists of:
- **Training examples**: Input-output grid pairs showing the pattern
- **Test input**: A new grid requiring transformation
- **Goal**: Produce the correct output grid

### Grid Specifications

| Property | Value |
|----------|-------|
| Grid size | Variable (up to 30x30) |
| Colors | 10 discrete values (0-9) |
| Output | Must match exactly (all cells) |
| Attempts | 2 tries per test input |

### Files

| File | Description |
|------|-------------|
| `arc-agi_training-challenges.json` | Training task inputs/outputs |
| `arc-agi_training-solutions.json` | Training task solutions |
| `arc-agi_evaluation-challenges.json` | Evaluation task inputs |
| `arc-agi_evaluation-solutions.json` | Evaluation solutions (hidden) |

### Task Types (Examples)

| Pattern Type | Description |
|--------------|-------------|
| Object manipulation | Move, rotate, scale objects |
| Pattern completion | Fill in missing parts |
| Color transformations | Change colors based on rules |
| Counting/arithmetic | Numerical operations on grids |
| Symmetry | Mirror, rotate, reflect |
| Flood fill | Fill regions based on rules |

---

## Evaluation

**Metric**: Task Solve Rate

A task is "solved" if the predicted output grid exactly matches the expected answer.

```
Score = (Tasks correctly solved) / (Total tasks)
```

### Important Notes

- Output grid dimensions must be correct
- Every cell must match (no partial credit)
- 2 attempts allowed per test input
- Both training and evaluation sets used

---

## Prize Structure

| Achievement | Prize |
|-------------|-------|
| 1st-5th Place | Share of $100,000 |
| Beat 85% threshold | Additional $500,000 |

**Breakthrough Prize**: $500,000 for achieving >85% on the private evaluation set.

---

## Top Solutions

### 1st Place - MindsAI

**Key Approach**:
- Hybrid neural-symbolic system
- Test-time fine-tuning of language models
- Program synthesis with domain-specific language
- Ensemble of diverse reasoning approaches

### 2nd Place - Omni-ARC

**Author**: Guillermo Barbadillo

**Technique**:
- Multi-strategy approach ("Omni")
- LLM-based reasoning
- Program search with DSL
- Pattern-specific solvers
- Achieved significant improvement over baselines

### 3rd Place Solution

**Author**: alijs

**Innovation**:
- Efficient program synthesis
- Custom domain-specific language
- Pruning strategies for search space

### 4th Place Solution

**Author**: William Wu Chengyuan

**Approach**:
- Neural network augmented search
- Pattern recognition preprocessing
- Efficient enumeration strategies

### 5th Place Solution

**Author**: gromml

**Technique**:
- Specialized solvers for task categories
- Efficient test-time computation
- Ensemble of approaches

---

## Common Winning Strategies

### 1. Domain-Specific Languages (DSL)

Define primitive operations and compose them:

```python
# Example DSL primitives
primitives = {
    'rotate': lambda grid, n: np.rot90(grid, n),
    'flip_h': lambda grid: np.fliplr(grid),
    'flip_v': lambda grid: np.flipud(grid),
    'extract_color': lambda grid, c: (grid == c).astype(int),
    'flood_fill': lambda grid, pos, color: flood_fill(grid, pos, color),
    'scale': lambda grid, factor: np.kron(grid, np.ones((factor, factor))),
}

# Program = sequence of operations
program = ['extract_color(5)', 'rotate(1)', 'scale(2)']
```

### 2. Test-Time Training

Fine-tune models on the few training examples:

```python
def solve_task(task):
    # Extract training examples
    train_pairs = task['train']

    # Fine-tune model on task-specific examples
    model = fine_tune_on_examples(base_model, train_pairs)

    # Apply to test input
    test_input = task['test'][0]['input']
    prediction = model.predict(test_input)

    return prediction
```

### 3. Program Synthesis with Search

```python
def synthesize_program(train_examples, max_depth=5):
    queue = [([], train_examples)]

    while queue:
        program, examples = queue.pop(0)

        if len(program) > max_depth:
            continue

        if validates_all(program, examples):
            return program

        for primitive in primitives:
            new_program = program + [primitive]
            if not prunable(new_program, examples):
                queue.append((new_program, examples))

    return None
```

### 4. LLM-Augmented Reasoning

```python
def llm_solve(task):
    prompt = f"""
    Given these input-output examples:
    {format_examples(task['train'])}

    What transformation rule produces the output from input?
    Apply this rule to: {task['test']['input']}
    """

    response = llm.generate(prompt)
    grid = parse_grid_from_response(response)
    return grid
```

### 5. Ensemble of Approaches

Combine multiple solving strategies:

| Strategy | Strength |
|----------|----------|
| DSL search | Precise, interpretable |
| Neural network | Pattern recognition |
| LLM reasoning | Abstract understanding |
| Hand-coded rules | Fast, reliable |

---

## Technical Insights

### Why ARC is Hard

| Challenge | Description |
|-----------|-------------|
| Novelty | Tasks never seen during training |
| Few-shot | Only 2-5 examples per task |
| Exact matching | No partial credit |
| Search space | Exponential program combinations |
| Abstraction | Must infer high-level concepts |

### Key Observations from Winners

1. **No single approach dominates** - Ensembles essential
2. **Test-time computation** - Spend compute at inference
3. **Task-specific adaptation** - Fine-tune per task
4. **DSL design** - Choice of primitives matters
5. **Efficient search** - Pruning is critical

### Human vs AI Performance

| Group | Score |
|-------|-------|
| Average human | ~85% |
| Pre-competition AI | ~34% |
| Winning solutions | ~53% |
| OpenAI o3 (post-comp) | ~87.5% |

### The 85% Threshold

The $500K breakthrough prize was NOT claimed during the competition:
- Requires >85% on private evaluation
- Even winning solution achieved ~53%
- Demonstrates the difficulty of true generalization

---

## Notable Developments

### Post-Competition Progress

- **OpenAI o3** reportedly achieved 87.5% (December 2024)
- Suggests compute-intensive approaches may succeed
- Debate continues about what counts as "efficient learning"

### Ensemble Insight

An ensemble of all Kaggle submissions scored 81%:
- Diversity of approaches valuable
- Combined solutions approach human level
- Individual systems still limited

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | 12 hours total |
| Internet | Disabled |
| GPU | Available |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 2nd | [Discussion](https://www.kaggle.com/c/arc-prize-2024/discussion/553765) |
| 3rd | [Discussion](https://www.kaggle.com/c/arc-prize-2024/discussion/553830) |
| 4th | [Discussion](https://www.kaggle.com/c/arc-prize-2024/discussion/553787) |
| 5th | [Discussion](https://www.kaggle.com/c/arc-prize-2024/discussion/553791) |

---

## Resources

- [ARCPrize.org](https://arcprize.org) - Official competition site
- [Interactive ARC Explorer](https://arcprize.org/arc) - Explore tasks visually
- [ARC-AGI GitHub](https://github.com/fchollet/ARC-AGI) - Original benchmark

---

## Citation

```bibtex
@misc{arc-prize-2024,
    author = {ARC Prize Foundation},
    title = {ARC Prize 2024},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/arc-prize-2024}},
    note = {Kaggle, ARCPrize.org}
}

@article{chollet2019measure,
    title={On the Measure of Intelligence},
    author={Chollet, Fran{\c{c}}ois},
    journal={arXiv preprint arXiv:1911.01547},
    year={2019}
}
```
