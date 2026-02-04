# Santa 2024 - The Perplexity Permutation Puzzle

> Help Rudolph descramble holiday-related words to make the LLMs happy!

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Combinatorial Optimization / Permutation |
| **Data Domain** | NLP / LLM Perplexity |
| **ML Approach** | Search Algorithms + LLM Scoring |
| **Key Techniques** | Beam Search, Simulated Annealing, Genetic Algorithms |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Optimization Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,514 |
| **Timeline** | December 2024 - January 2025 |
| **Evaluation Metric** | Santa 2024 Metric (Perplexity-based) |
| **Host** | Kaggle |

## Problem Description

Find permutations of scrambled holiday-related words that minimize the perplexity score when evaluated by LLMs.

### The Challenge

- Scrambled text needs to be descrambled
- LLM perplexity used as optimization target
- Combinatorial explosion of possible permutations
- Must find globally optimal arrangements

### The Santa Competition Tradition

Kaggle's annual Santa competition is a beloved tradition featuring:
- Holiday-themed optimization puzzles
- Creative problem formulations
- Community collaboration and competition

---

## Data Description

### Task Structure

| Component | Description |
|-----------|-------------|
| Scrambled texts | Holiday words/phrases in wrong order |
| Perplexity model | LLM that scores arrangements |
| Target | Minimize perplexity (natural ordering) |

### Example

Scrambled: `"tree Christmas the decorated we"`
Target: `"we decorated the Christmas tree"`
Score: Lower perplexity for grammatical text

---

## Evaluation

**Metric**: Santa 2024 Metric (Perplexity-based)

Perplexity measures how "surprised" the LLM is by the text:
- Lower perplexity = more natural/expected text
- Goal: Find permutations with minimum perplexity

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
- Hybrid search combining multiple algorithms
- Efficient perplexity caching
- Parallel exploration of search space

### 2nd Place Solution

**Technique**:
- Simulated annealing with smart restarts
- Local search with swap operations
- Temperature schedule optimization

### 3rd Place Solution

**Innovation**:
- Genetic algorithm with crossover operators
- Population diversity maintenance
- Elite preservation strategies

---

## Common Winning Strategies

### 1. Search Algorithms

| Algorithm | Strengths |
|-----------|-----------|
| Beam Search | Maintains multiple candidates |
| Simulated Annealing | Escapes local optima |
| Genetic Algorithm | Population-based exploration |
| Branch & Bound | Optimal for small instances |

### 2. Perplexity Computation

Efficient evaluation is critical:
```python
def compute_perplexity(text, model):
    tokens = tokenize(text)
    log_probs = model.score(tokens)
    return exp(-mean(log_probs))
```

### 3. Move Operators

| Operator | Description |
|----------|-------------|
| Swap | Exchange two elements |
| Insert | Move element to new position |
| Reverse | Reverse subsequence |
| Block move | Move contiguous block |

### 4. Caching Strategies

Avoid redundant computation:
```python
cache = {}
def cached_perplexity(permutation):
    key = tuple(permutation)
    if key not in cache:
        cache[key] = compute_perplexity(permutation)
    return cache[key]
```

### 5. Initialization Strategies

Start from good solutions:
- Greedy construction
- Frequency-based ordering
- N-gram guided initialization

---

## Technical Insights

### Permutation Search Space

For n elements:
- n! possible permutations
- Exponential growth
- Exact solution intractable for large n

### Local vs Global Search

| Approach | Trade-off |
|----------|-----------|
| Local search | Fast but may get stuck |
| Global search | Thorough but slow |
| Hybrid | Best of both worlds |

### Parallelization

```python
from multiprocessing import Pool

def parallel_search(candidates):
    with Pool(processes=num_cpus) as pool:
        scores = pool.map(evaluate, candidates)
    return scores
```

### LLM-Specific Considerations

- Tokenization affects scoring
- Context window limits
- Batch evaluation for efficiency
- Model-specific perplexity behavior

### Greedy Pitfalls

Local greedy choices may not be globally optimal:
- "the" might score well initially
- But better to place it before nouns
- Lookahead helps avoid traps

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/santa-2024/discussion/560560) |
| 2nd | [Discussion](https://www.kaggle.com/c/santa-2024/discussion/560540) |
| 3rd | [Discussion](https://www.kaggle.com/c/santa-2024/discussion/560620) |
| 4th | [Discussion](https://www.kaggle.com/c/santa-2024/discussion/560536) |
| 5th | [Discussion](https://www.kaggle.com/c/santa-2024/discussion/560597) |

---

## Citation

```bibtex
@misc{santa-2024,
    title = {Santa 2024 - The Perplexity Permutation Puzzle},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/santa-2024}},
    note = {Kaggle}
}
```
