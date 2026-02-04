# AI Mathematical Olympiad - Progress Prize 1

> Solve national-level math challenges using artificial intelligence models

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Mathematical Reasoning / Question Answering |
| **Data Domain** | Mathematics / Olympiad Problems |
| **ML Approach** | LLM Fine-tuning + Code Execution |
| **Key Techniques** | Chain-of-Thought, Self-Consistency, Tool Use |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $1,048,576 |
| **Teams** | 1,161 |
| **Timeline** | April - June 2024 |
| **Evaluation Metric** | Accuracy Score |
| **Host** | AIMO (AI Mathematical Olympiad) |

## Problem Description

Create algorithms that can solve novel mathematical problems written in LaTeX, advancing AI's mathematical reasoning capabilities.

### The Challenge

- Solve 110 novel math problems
- Problems at intermediate to national olympiad level
- Answers are integers
- 2 attempts per problem
- Must generalize to unseen problem types

### Why It Matters

- **AGI Progress**: Mathematical reasoning is a key milestone
- **Novel Problems**: Tests true reasoning, not memorization
- **$10M Grand Prize**: Part of larger AIMO initiative
- **Open Development**: Advance open-source math AI

---

## Data Description

### Problem Characteristics

| Property | Description |
|----------|-------------|
| Format | LaTeX mathematical notation |
| Difficulty | National olympiad level |
| Topics | Algebra, geometry, number theory, combinatorics |
| Answers | Integer values |
| Novel | Created specifically to avoid train-test leakage |

### Example Problem Format

```latex
Let $n$ be a positive integer such that $n^2 + 12n - 2007$
is a perfect square. Find the sum of all such $n$.
```

### Benchmark

The Gemma 7B baseline achieves 3/50 on test problems, demonstrating the difficulty level.

---

## Evaluation

**Metric**: Accuracy Score

```
Score = Number of correct answers / Total problems
```

Each problem allows 2 attempts. Getting either attempt correct counts as solved.

---

## Prize Structure

| Prize | Amount |
|-------|--------|
| Competition Pool | $1,048,576 |
| Grand Prize (IMO Gold) | Up to $5,000,000 |

Part of the $10M AIMO Prize initiative.

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Fine-tuned mathematical LLMs
- Chain-of-thought prompting
- Multiple solution attempts
- Code execution for verification

### 2nd Place Solution

**Technique**:
- DeepSeekMath model
- Self-consistency sampling
- Tool-integrated reasoning

### 3rd Place Solution

**Innovation**:
- Ensemble of math-specialized models
- Python code generation for computation
- Symbolic manipulation

---

## Common Winning Strategies

### 1. Chain-of-Thought Prompting

```python
def solve_problem(problem_text, model):
    prompt = f"""
    Solve the following mathematical problem step by step.
    Show all your work and reasoning clearly.
    At the end, provide the final integer answer.

    Problem:
    {problem_text}

    Solution:
    Let me work through this step by step.
    """

    response = model.generate(prompt, max_tokens=2048)
    answer = extract_final_answer(response)
    return answer
```

### 2. Self-Consistency Sampling

```python
def solve_with_self_consistency(problem, model, n_samples=10):
    answers = []

    for _ in range(n_samples):
        # Generate different reasoning paths
        response = model.generate(
            problem,
            temperature=0.7,  # Add randomness
            max_tokens=2048
        )
        answer = extract_answer(response)
        if answer is not None:
            answers.append(answer)

    # Majority vote
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return None
```

### 3. Code Execution Integration

```python
def solve_with_code(problem, model):
    prompt = f"""
    Solve this math problem by writing Python code.
    The code should compute and print the final answer.

    Problem: {problem}

    Python code:
    ```python
    """

    code = model.generate(prompt)

    # Execute code safely
    try:
        result = safe_execute(code)
        return int(result)
    except:
        return None
```

### 4. Mathematical Model Selection

Key models used:
| Model | Specialization |
|-------|----------------|
| DeepSeekMath | Mathematical reasoning |
| Llemma | Math/science |
| Code Llama | Code generation |
| Mistral + Math finetune | General + math |

### 5. Two-Attempt Strategy

```python
def solve_two_attempts(problem, model):
    # First attempt: Direct reasoning
    answer1 = solve_with_reasoning(problem, model)

    # Second attempt: Code-based
    answer2 = solve_with_code(problem, model)

    # Return both for submission
    return [answer1, answer2]
```

---

## Technical Insights

### Mathematical Problem Types

| Type | Example |
|------|---------|
| Algebra | Polynomial equations, inequalities |
| Number Theory | Divisibility, primes, modular arithmetic |
| Geometry | Constructions, area, angles |
| Combinatorics | Counting, permutations |

### Why Math is Hard for LLMs

| Challenge | Description |
|-----------|-------------|
| Multi-step reasoning | Long chains of logic |
| Precision | Exact answers required |
| Abstraction | Variable manipulation |
| Verification | Hard to check intermediate steps |

### Avoiding Train-Test Leakage

Competition organizers:
- Created 110 novel problems
- Problems not on the internet
- International team of problem creators
- Various difficulty levels

### DeepSeekMath Eligibility

Discussion arose about whether certain models were eligible:
- Must be open-source
- Must be reproducible
- Competition rules clarified

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | 9 hours |
| Internet | Disabled |
| GPU | P100 |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/519303) |
| 2nd | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/518964) |
| 3rd | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/517206) |
| 4th | [Discussion](https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/518960) |

---

## Resources

- [AIMO Prize](https://aimoprize.com/) - Full $10M initiative
- [IMO Problems](https://www.imo-official.org/) - Reference difficulty

---

## Citation

```bibtex
@misc{ai-mathematical-olympiad-prize,
    author = {AIMO},
    title = {AI Mathematical Olympiad - Progress Prize 1},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/ai-mathematical-olympiad-prize}},
    note = {Kaggle}
}
```
