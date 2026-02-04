# LMSYS - Chatbot Arena Human Preference Predictions

> Predicting Human Preferences in the Wild

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Class Classification |
| **Data Domain** | NLP / LLM Evaluation |
| **ML Approach** | Transformer Fine-tuning + Embeddings |
| **Key Techniques** | RLHF Data, Pairwise Comparison, Preference Modeling |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Competition |
| **Total Prize** | $100,000 |
| **Teams** | 1,849 |
| **Timeline** | 2024 |
| **Evaluation Metric** | Log Loss |
| **Host** | LMSYS |

## Problem Description

Predict which LLM response humans will prefer when shown two responses to the same prompt. This models the core evaluation task behind RLHF (Reinforcement Learning from Human Feedback).

### The Challenge

- Given: prompt + two LLM responses (A and B)
- Predict: probability distribution over [A wins, B wins, Tie]
- Data: Real human preferences from Chatbot Arena
- Goal: Model human preference behavior

### Why It Matters

- **RLHF**: Human preference is core to aligning LLMs
- **Evaluation**: Better preference models improve AI safety
- **Research**: Understand what makes responses "better"
- **Scalability**: Replace expensive human annotation

---

## Data Description

### Chatbot Arena

LMSYS Chatbot Arena is a crowdsourced platform where:
- Users submit prompts to two anonymous LLMs
- Users choose which response they prefer
- Votes create ranking of LLM quality

### Data Format

| Field | Description |
|-------|-------------|
| prompt | User's input question/task |
| response_a | First model's response |
| response_b | Second model's response |
| model_a | Identity of first model (hidden in test) |
| model_b | Identity of second model (hidden in test) |
| winner | Human choice: A, B, or tie |

### Outcome Classes

| Class | Description |
|-------|-------------|
| winner_model_a | Human preferred response A |
| winner_model_b | Human preferred response B |
| winner_tie | Human saw no clear winner |

---

## Evaluation

**Metric**: Log Loss (Multi-class)

```python
log_loss = -1/N * Σ Σ y_ij * log(p_ij)
```

Where predictions are probabilities for each of three outcomes.

### Why Log Loss?

- Measures calibration of predicted probabilities
- Penalizes confident wrong predictions heavily
- Appropriate for probabilistic predictions

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $100,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Fine-tuned large language models
- Ensemble of multiple model sizes
- Careful handling of prompt/response structure
- Data augmentation by swapping A/B

### 2nd Place Solution

**Technique**:
- DeBERTa-based classifiers
- Feature engineering on response properties
- Length normalization
- Model ID prediction as auxiliary task

### 3rd Place Solution

**Innovation**:
- Embedding-based approach
- Contrastive learning
- Cross-encoder architecture

### 5th Place Solution

**Approach**:
- LLM-as-judge features
- Metadata features (length, formatting)
- Gradient boosting on embeddings

---

## Common Winning Strategies

### 1. Transformer Fine-tuning

```python
from transformers import AutoModelForSequenceClassification

class PreferenceModel(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-large'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# Input format: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
```

### 2. Data Augmentation by Swapping

```python
def augment_by_swapping(df):
    """Swap A and B to double data, adjust labels"""
    swapped = df.copy()
    swapped['response_a'] = df['response_b']
    swapped['response_b'] = df['response_a']

    # Swap labels
    label_map = {
        'winner_model_a': 'winner_model_b',
        'winner_model_b': 'winner_model_a',
        'winner_tie': 'winner_tie'
    }
    swapped['winner'] = df['winner'].map(label_map)

    return pd.concat([df, swapped], ignore_index=True)
```

### 3. Feature Engineering

```python
def extract_features(row):
    features = {}

    # Length features
    features['len_a'] = len(row['response_a'])
    features['len_b'] = len(row['response_b'])
    features['len_ratio'] = features['len_a'] / (features['len_b'] + 1)

    # Formatting features
    features['code_blocks_a'] = row['response_a'].count('```')
    features['code_blocks_b'] = row['response_b'].count('```')

    features['bullet_points_a'] = row['response_a'].count('\n-')
    features['bullet_points_b'] = row['response_b'].count('\n-')

    # Complexity features
    features['avg_word_len_a'] = np.mean([len(w) for w in row['response_a'].split()])
    features['avg_word_len_b'] = np.mean([len(w) for w in row['response_b'].split()])

    return features
```

### 4. Cross-Encoder Architecture

```python
class CrossEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Separate processing for each response
        self.prompt_attention = nn.MultiheadAttention(768, 8)
        self.comparison_layer = nn.Linear(768 * 2, 768)
        self.classifier = nn.Linear(768, 3)

    def forward(self, prompt_ids, resp_a_ids, resp_b_ids):
        prompt_emb = self.encoder(prompt_ids).last_hidden_state[:, 0]
        resp_a_emb = self.encoder(resp_a_ids).last_hidden_state[:, 0]
        resp_b_emb = self.encoder(resp_b_ids).last_hidden_state[:, 0]

        # Combine embeddings
        combined = torch.cat([resp_a_emb, resp_b_emb], dim=1)
        compared = self.comparison_layer(combined)

        return self.classifier(compared)
```

### 5. LLM-as-Judge Features

```python
def get_llm_judge_score(prompt, response):
    """Use another LLM to evaluate response quality"""
    judge_prompt = f"""
    Rate the following response to the given prompt on a scale of 1-10.

    Prompt: {prompt}
    Response: {response}

    Consider:
    - Relevance to the prompt
    - Accuracy of information
    - Clarity and coherence
    - Helpfulness

    Score (1-10):
    """
    score = llm.generate(judge_prompt)
    return parse_score(score)
```

---

## Technical Insights

### What Humans Prefer

Research shows humans tend to prefer:
| Factor | Impact |
|--------|--------|
| Longer responses | Positive (to a point) |
| Formatting (lists, code blocks) | Positive |
| Confident tone | Positive |
| Accuracy | Strong positive |
| Relevance | Strong positive |

### Position Bias

Humans may have slight preference for first response - must handle:
```python
# Account for position bias
bias_term = learned_position_bias  # trained parameter
logits_a = logits_a - bias_term
logits_b = logits_b + bias_term
```

### Tie Prediction

Ties are challenging:
- Humans often default to tie when uncertain
- True ties (equal quality) are rare
- Model should predict tie probability carefully

### Model Size vs Preference

Generally:
- Larger models win more often
- But small models can win on specific tasks
- Model identity not available in test set

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/527629) |
| 2nd | [Discussion](https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/527685) |
| 3rd | [Discussion](https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/527766) |
| 4th | [Discussion](https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/529067) |
| 5th | [Discussion](https://www.kaggle.com/c/lmsys-chatbot-arena/discussion/527669) |

---

## Resources

- [LMSYS Chatbot Arena](https://chat.lmsys.org/)
- [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

---

## Citation

```bibtex
@misc{lmsys-chatbot-arena,
    author = {LMSYS},
    title = {LMSYS - Chatbot Arena Human Preference Predictions},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/lmsys-chatbot-arena}},
    note = {Kaggle}
}
```
