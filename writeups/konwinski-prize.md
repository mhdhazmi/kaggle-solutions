# Konwinski Prize

> $1M for the AI that can close 90% of new GitHub issues

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Code Generation / Automated Software Engineering |
| **Data Domain** | Software Engineering / GitHub Issues |
| **ML Approach** | Large Language Models (Agentic AI) |
| **Key Techniques** | Code Understanding, Issue Resolution, Automated Testing |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Research Competition |
| **Total Prize** | $1,225,000 |
| **Teams** | 617 |
| **Timeline** | 2025 |
| **Evaluation Metric** | Issue Resolution Rate |
| **Host** | Konwinski Prize Foundation |

## Problem Description

Build an AI system that can automatically resolve GitHub issues. The grand prize requires achieving a **90% resolution rate** on new, unseen issues.

### The Challenge

- Read and understand GitHub issues
- Analyze the relevant codebase
- Generate appropriate code fixes
- Create pull requests that pass tests
- Handle diverse issue types (bugs, features, documentation)

### Why It Matters

- Accelerate software development
- Reduce developer workload
- Enable more accessible software maintenance
- Push boundaries of AI code understanding

---

## Evaluation

**Metric**: Issue Resolution Rate

An issue is considered "resolved" when:
1. The AI-generated fix addresses the issue
2. All existing tests pass
3. The fix doesn't introduce regressions
4. Code quality meets standards

---

## Prize Structure

| Achievement | Prize |
|-------------|-------|
| 90% Resolution Rate | $1,000,000+ |
| 1st Place | TBD |
| 2nd Place | TBD |
| 3rd Place | TBD |
| ... | ... |

**Total Prize Pool**: $1,225,000

---

## Top Solutions

### 1st Place: Eduardo Rocha de Andrade

**Key Approach**:
- Multi-agent architecture for issue understanding
- Codebase analysis and context retrieval
- Iterative fix generation with test validation
- Pull request automation

*(Full details in linked writeup)*

### 2nd Place: Camaro (Public 2nd)

*(Details in linked writeup)*

### 3rd Place Solution

*(Details in linked writeup)*

---

## Common Winning Strategies

### 1. Multi-Agent Systems
Breaking down the task into specialized agents:
- Issue analyzer
- Code navigator
- Fix generator
- Test runner
- PR creator

### 2. Codebase Understanding
Deep analysis of repository structure, dependencies, and coding patterns.

### 3. Iterative Refinement
Generate fix → Run tests → Analyze failures → Refine fix → Repeat

### 4. Context Retrieval
Efficiently retrieving relevant code context for large repositories.

### 5. Test-Driven Development
Using existing tests to validate fixes and catch regressions.

---

## Technical Insights

### Key Challenges

1. **Codebase scale**: Large repos with millions of lines of code
2. **Issue ambiguity**: Natural language descriptions can be unclear
3. **Test coverage**: Not all issues have comprehensive tests
4. **Side effects**: Fixes may introduce unintended changes

### Typical Pipeline

```
Issue → Parse & Understand → Locate Relevant Code → Generate Fix
    → Run Tests → Validate → Create PR
```

### Agent Architecture

```
┌─────────────────┐
│  Issue Parser   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Code Navigator  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Fix Generator  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Test Runner    │
└────────┬────────┘
         ↓
┌─────────────────┐
│   PR Creator    │
└─────────────────┘
```

---

## Technical Requirements

Submissions must:
- Operate autonomously on new issues
- Not require human intervention
- Complete within reasonable time limits
- Handle diverse repository types

---

## Solution Links

| Place | Team | Solution |
|-------|------|----------|
| 1st | Eduardo Rocha de Andrade | [Writeup](https://www.kaggle.com/c/konwinski-prize/writeups/eduardo-rocha-de-andrade-1st-place-solution-write-) |
| 2nd | Camaro | [Writeup](https://www.kaggle.com/c/konwinski-prize/writeups/camaro-public-2nd-place-solution) |
| 3rd | - | [Writeup](https://www.kaggle.com/c/konwinski-prize/writeups/3rd-place-solution) |
| 4th | - | [Writeup](https://www.kaggle.com/c/konwinski-prize/writeups/4th-place-solution) |
| 5th | - | [Writeup](https://www.kaggle.com/c/konwinski-prize/writeups/5th-place-solution) |

---

## Citation

```bibtex
@misc{konwinski-prize,
    title = {Konwinski Prize},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/konwinski-prize}},
    note = {Kaggle}
}
```
