# LLM Prompt Recovery

> Recover the prompt used to transform a given text

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Text Generation / Prompt Reconstruction |
| **Data Domain** | NLP / LLM Prompting |
| **ML Approach** | LLM Inference + Embedding Similarity |
| **Key Techniques** | Prompt Engineering, Synthetic Data Generation, Cosine Similarity |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $200,000 |
| **Teams** | 2,175 |
| **Timeline** | February - April 2024 |
| **Evaluation Metric** | Sharpened Cosine Similarity |
| **Host** | Kaggle |

## Problem Description

Recover the LLM prompt that was used to rewrite a given text - an inverse problem that tests understanding of how LLMs respond to prompts.

### The Challenge

- Given original text and rewritten version
- Determine what prompt was given to Gemma 7B-it
- Match semantic meaning of original prompt
- Handle diverse rewriting styles

### Why It Matters

- **LLM Understanding**: Reveals how prompts affect output
- **Security**: Prompt injection detection
- **Interpretability**: Understanding LLM behavior
- **Reverse Engineering**: Recovering system prompts

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `train.csv` | Single example with original, prompt, and rewritten text |
| `test.csv` | Original and rewritten texts (prompts hidden) |
| `sample_submission.csv` | Submission format |

### Data Fields

| Column | Description |
|--------|-------------|
| `id` | Unique identifier |
| `original_text` | The source text before rewriting |
| `rewrite_prompt` | Target: the prompt used (train only) |
| `rewritten_text` | Output from Gemma 7B-it |

### Key Insight

Only ONE training example provided - competitors needed to generate synthetic data to train models.

---

## Evaluation

**Metric**: Sharpened Cosine Similarity (SCS)

```python
from sentence_transformers import SentenceTransformer

def sharpened_cosine_similarity(pred_prompt, true_prompt, exponent=3):
    """
    Uses sentence-t5-base embeddings
    Sharpening with exponent=3 penalizes poor matches heavily
    """
    model = SentenceTransformer('sentence-t5-base')

    pred_emb = model.encode(pred_prompt)
    true_emb = model.encode(true_prompt)

    # Cosine similarity
    cos_sim = np.dot(pred_emb, true_emb) / (
        np.linalg.norm(pred_emb) * np.linalg.norm(true_emb)
    )

    # Sharpen with exponent
    scs = cos_sim ** exponent

    return scs
```

Higher is better. The sharpening penalizes semantically distant predictions.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $100,000 |
| 2nd | $40,000 |
| 3rd | $20,000 |
| 4th | $14,000 |
| 5th | $11,000 |
| 6th | $10,000 |
| 7th | $5,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Generated massive synthetic training data
- Fine-tuned LLMs to predict prompts
- Ensemble of multiple model sizes
- Careful prompt template analysis

### 2nd Place Solution

**Technique**:
- Analyzed Gemma's prompt format patterns
- Template-based prompt generation
- Similarity-based ranking

### 3rd Place Solution

**Innovation**:
- Few-shot learning approach
- Prompt retrieval from candidate set
- Embedding space optimization

---

## Common Winning Strategies

### 1. Synthetic Data Generation

```python
def generate_synthetic_data(original_texts, num_prompts=1000):
    """Generate diverse rewrite prompts and outputs"""
    prompt_templates = [
        "Rewrite this text in the style of {}",
        "Make this text more {}",
        "Transform this passage to be {}",
        "Rewrite using {} tone",
        "Convert to {} format"
    ]

    styles = ['formal', 'casual', 'academic', 'poetic', 'humorous',
              'professional', 'simple', 'dramatic', 'concise', 'elaborate']

    training_data = []
    for text in original_texts:
        for template in prompt_templates:
            for style in styles:
                prompt = template.format(style)
                # Generate rewrite using Gemma
                rewritten = gemma_rewrite(text, prompt)
                training_data.append({
                    'original': text,
                    'prompt': prompt,
                    'rewritten': rewritten
                })

    return training_data
```

### 2. LLM for Prompt Prediction

```python
def predict_prompt_with_llm(original_text, rewritten_text):
    """Use LLM to predict the rewrite prompt"""
    inference_prompt = f"""
    Original text: {original_text}

    Rewritten text: {rewritten_text}

    What prompt was likely used to transform the original text
    into the rewritten version? Provide only the prompt, nothing else.
    """

    predicted_prompt = llm.generate(inference_prompt)
    return predicted_prompt
```

### 3. Template Matching

```python
def match_prompt_template(original, rewritten, templates):
    """Match against known prompt patterns"""
    scores = []

    for template in templates:
        # Generate expected output with this template
        expected = generate_with_template(original, template)

        # Compare to actual rewritten text
        similarity = compute_text_similarity(expected, rewritten)
        scores.append((template, similarity))

    # Return best matching template
    best_template = max(scores, key=lambda x: x[1])[0]
    return best_template
```

### 4. Embedding-Based Retrieval

```python
from sentence_transformers import SentenceTransformer

class PromptRetriever:
    def __init__(self, candidate_prompts):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.prompts = candidate_prompts
        self.embeddings = self.model.encode(candidate_prompts)

    def retrieve(self, original, rewritten, top_k=5):
        # Create query from text pair
        query = f"Original: {original[:200]} Rewritten: {rewritten[:200]}"
        query_emb = self.model.encode(query)

        # Find similar prompts
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.prompts[i] for i in top_indices]
```

### 5. Style Analysis Features

```python
def extract_style_features(original, rewritten):
    """Extract features indicating style transformation"""
    features = {}

    # Length change
    features['length_ratio'] = len(rewritten) / len(original)

    # Vocabulary complexity
    features['orig_avg_word_len'] = np.mean([len(w) for w in original.split()])
    features['rewr_avg_word_len'] = np.mean([len(w) for w in rewritten.split()])

    # Punctuation changes
    features['orig_exclaim'] = original.count('!')
    features['rewr_exclaim'] = rewritten.count('!')

    # Sentence structure
    features['orig_sentences'] = len(original.split('.'))
    features['rewr_sentences'] = len(rewritten.split('.'))

    # Formality indicators
    formal_words = {'therefore', 'however', 'furthermore', 'consequently'}
    features['formality_increase'] = (
        sum(1 for w in rewritten.lower().split() if w in formal_words) -
        sum(1 for w in original.lower().split() if w in formal_words)
    )

    return features
```

---

## Technical Insights

### Sharpened Cosine Similarity Effect

| Cosine Sim | SCS (exp=3) | Interpretation |
|------------|-------------|----------------|
| 1.0 | 1.0 | Perfect match |
| 0.9 | 0.729 | Good match |
| 0.8 | 0.512 | Moderate |
| 0.7 | 0.343 | Weak |
| 0.5 | 0.125 | Poor |

### Common Rewrite Patterns

| Pattern Type | Example Prompt |
|--------------|----------------|
| Style transfer | "Rewrite in the style of Shakespeare" |
| Tone change | "Make this more formal/casual" |
| Format change | "Convert to bullet points" |
| Simplification | "Explain like I'm five" |
| Elaboration | "Expand with more detail" |

### Why This Was Hard

1. **Inverse problem**: Many prompts could produce similar outputs
2. **Semantic equivalence**: Different phrasings mean the same thing
3. **Limited training data**: Only one example provided
4. **Black box model**: Gemma's behavior not fully predictable

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | 9 hours |
| GPU Runtime | 9 hours |
| Internet | Disabled |
| External Data | Allowed (including pre-trained models) |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494344) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494279) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494360) |

---

## Citation

```bibtex
@misc{llm-prompt-recovery,
    author = {Kaggle},
    title = {LLM Prompt Recovery},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/llm-prompt-recovery}},
    note = {Kaggle}
}
```
