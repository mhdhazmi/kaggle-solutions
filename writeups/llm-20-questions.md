# LLM 20 Questions

> Guess the secret word in this cooperative game of question asking and answering

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Agent Game / NLP |
| **Data Domain** | Language Games / LLM Interaction |
| **ML Approach** | LLM Prompting + Search Strategies |
| **Key Techniques** | Question Generation, Word Elimination, Game Theory |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Competition |
| **Total Prize** | $50,000 |
| **Teams** | 832 |
| **Timeline** | 2024 |
| **Evaluation Metric** | Custom (Game Success Rate) |
| **Host** | Kaggle |

## Problem Description

Build LLM-based agents to play the classic "20 Questions" game - one agent asks yes/no questions, another answers, and the goal is to guess a secret word within 20 questions.

### The Challenge

- **Questioner Agent**: Ask strategic yes/no questions
- **Answerer Agent**: Respond accurately to questions about secret word
- **Guesser**: Identify the secret word efficiently
- **Cooperative**: Both agents work toward same goal

### Why It Matters

- **LLM Evaluation**: Tests reasoning and world knowledge
- **Multi-Agent Systems**: Coordination between AI systems
- **Information Theory**: Optimal question strategies
- **Game AI**: Beyond adversarial to cooperative

---

## Data Description

### Game Format

| Component | Description |
|-----------|-------------|
| Secret word | Hidden word to be guessed |
| Questions | Yes/no questions about the word |
| Answers | Binary responses |
| Turns | Maximum 20 questions |
| Win condition | Correctly guess the secret word |

### Word Categories

Secret words span various categories:
- Objects
- Animals
- Places
- Concepts
- People

---

## Evaluation

**Metric**: Custom Game Success Rate

```
Score based on:
- Successful guesses
- Number of questions needed
- Question quality
- Answer accuracy
```

Agents are paired and play multiple games. Performance measured across all games.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Information-theoretic question selection
- Large candidate word pool
- Efficient elimination strategy
- Carefully crafted prompts

### 2nd Place Solution

**Technique**:
- Hierarchical questioning (category → specific)
- Pre-computed question trees
- Robust answer parsing

### 3rd Place Solution

**Innovation**:
- Ensemble of LLM strategies
- Fallback mechanisms
- Game state tracking

### 5th Place Solution

**Approach**:
- Binary search on word space
- Semantic clustering of candidates
- Dynamic question generation

---

## Common Winning Strategies

### 1. Information-Theoretic Questioning

```python
def select_best_question(candidates, questions):
    """Select question that maximizes expected information gain"""
    best_question = None
    best_info_gain = 0

    for question in questions:
        # Estimate P(yes) based on candidates
        yes_count = sum(1 for word in candidates if likely_yes(word, question))
        p_yes = yes_count / len(candidates)

        # Information gain maximized when p_yes = 0.5
        info_gain = binary_entropy(p_yes)

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_question = question

    return best_question

def binary_entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * log2(p) - (1-p) * log2(1-p)
```

### 2. Hierarchical Question Strategy

```python
# Start broad, then narrow
question_hierarchy = [
    # Level 1: Category
    "Is it a living thing?",
    "Is it something you can hold in your hand?",
    "Is it a place?",

    # Level 2: Subcategory
    "Is it an animal?",
    "Is it something you would find indoors?",

    # Level 3: Specific attributes
    "Does it have legs?",
    "Is it larger than a car?",

    # Level 4: Final identification
    "Is it commonly found in a kitchen?",
]
```

### 3. Questioner Agent Prompt

```python
questioner_prompt = """
You are playing 20 Questions. You need to guess a secret word by asking yes/no questions.

Current game state:
- Questions asked: {n_questions}/20
- Previous Q&A:
{qa_history}

Remaining candidate words: {n_candidates}

Your task: Ask a yes/no question that will best help narrow down the possibilities.
The question should:
1. Be answerable with yes or no
2. Eliminate roughly half the remaining possibilities
3. Be clear and unambiguous

Question:
"""
```

### 4. Answerer Agent Prompt

```python
answerer_prompt = """
You are answering questions in a game of 20 Questions.

The secret word is: {secret_word}

Question: {question}

Respond with exactly "yes" or "no" based on whether the question is true for the secret word.
Consider:
- Common knowledge about {secret_word}
- The most natural interpretation of the question
- Be consistent with previous answers

Answer:
"""
```

### 5. Candidate Elimination

```python
class QuestionEngine:
    def __init__(self, word_list):
        self.candidates = set(word_list)
        self.qa_history = []

    def update_candidates(self, question, answer):
        """Eliminate candidates inconsistent with answer"""
        self.qa_history.append((question, answer))

        new_candidates = set()
        for word in self.candidates:
            # Use LLM to check consistency
            if is_consistent(word, question, answer):
                new_candidates.add(word)

        self.candidates = new_candidates
        return len(self.candidates)

    def make_guess(self):
        """Guess when confident enough"""
        if len(self.candidates) <= 3:
            return self.candidates.pop()
        return None
```

---

## Technical Insights

### Optimal Strategy Theory

With perfect binary questions:
- 20 questions can distinguish 2^20 = ~1M words
- Real words are ~100K, so theoretically solvable
- But questions aren't perfectly binary in practice

### LLM Challenges

| Challenge | Impact |
|-----------|--------|
| Ambiguous questions | Answer uncertainty |
| LLM hallucination | Wrong answers |
| Semantic nuance | Interpretation varies |
| World knowledge | Must match LLM's knowledge |

### Question Categories by Information Gain

| Category | Typical Info Gain |
|----------|------------------|
| "Is it living?" | High (splits ~50%) |
| "Is it red?" | Low (very specific) |
| "Is it common?" | Medium (subjective) |
| "Does it have X?" | Variable |

### Multi-Agent Coordination

Both agents must:
1. Use same interpretation of questions
2. Have consistent world knowledge
3. Handle edge cases similarly
4. Avoid misunderstandings

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/llm-20-questions/discussion/531106) |
| 2nd | [Discussion](https://www.kaggle.com/c/llm-20-questions/discussion/529643) |
| 3rd | [Discussion](https://www.kaggle.com/c/llm-20-questions/discussion/531387) |
| 4th | [Discussion](https://www.kaggle.com/c/llm-20-questions/discussion/531883) |
| 5th | [Discussion](https://www.kaggle.com/c/llm-20-questions/discussion/531128) |

---

## Citation

```bibtex
@misc{llm-20-questions,
    title = {LLM 20 Questions},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/llm-20-questions}},
    note = {Kaggle}
}
```
