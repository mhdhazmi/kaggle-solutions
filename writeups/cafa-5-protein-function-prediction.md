# CAFA 5 Protein Function Prediction

> Predict the biological function of a protein

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Multi-Label Hierarchical Classification |
| **Data Domain** | Bioinformatics / Proteomics |
| **ML Approach** | Sequence Models + Graph Neural Networks |
| **Key Techniques** | Protein Language Models, Gene Ontology, Hierarchical Classification |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,625 |
| **Timeline** | April - December 2023 |
| **Evaluation Metric** | Weighted F-max Score |
| **Host** | Critical Assessment of Functional Annotation (CAFA) |

## Problem Description

Predict the biological functions of proteins based on their amino acid sequences - a fundamental challenge in computational biology.

### The Challenge

- Predict Gene Ontology (GO) terms for proteins
- Handle hierarchical structure of GO
- Multi-label: proteins have multiple functions
- Limited experimental annotations

### Gene Ontology Structure

| Subontology | Code | Description |
|-------------|------|-------------|
| Molecular Function | MF | What the protein does |
| Biological Process | BP | Processes it participates in |
| Cellular Component | CC | Where it's located |

### Why It Matters

- **Drug Discovery**: Understanding targets
- **Disease Research**: Linking proteins to pathology
- **Agriculture**: Improving crops
- **Basic Science**: Understanding life at molecular level

---

## Data Description

### Gene Ontology

GO is a directed acyclic graph (DAG):
- ~45,000 terms across 3 subontologies
- Hierarchical relationships (is_a, part_of)
- If a protein has term X, it also has all ancestors of X

### Files

| File | Description |
|------|-------------|
| `train_sequences.fasta` | Protein amino acid sequences |
| `train_terms.tsv` | GO term annotations |
| `go-basic.obo` | GO ontology structure |
| `train_taxonomy.tsv` | Species information |

### Data Properties

| Property | Value |
|----------|-------|
| Training proteins | ~140,000 |
| GO terms | ~45,000 |
| Sequence lengths | 50-35,000 amino acids |
| Annotations | Experimentally validated only |

---

## Evaluation

**Metric**: Weighted F-max Score

```python
def calculate_fmax(y_true, y_pred_probs, go_graph, thresholds=np.arange(0, 1, 0.01)):
    """
    F-max: maximum F1 over all thresholds
    Accounts for GO hierarchy (propagation)
    """
    f_scores = []

    for threshold in thresholds:
        # Binarize predictions
        y_pred = (y_pred_probs >= threshold).astype(int)

        # Propagate predictions through GO hierarchy
        y_pred_propagated = propagate_predictions(y_pred, go_graph)

        # Calculate precision and recall per protein
        precisions = []
        recalls = []

        for i in range(len(y_true)):
            true_terms = set(np.where(y_true[i])[0])
            pred_terms = set(np.where(y_pred_propagated[i])[0])

            if len(pred_terms) > 0:
                precision = len(true_terms & pred_terms) / len(pred_terms)
            else:
                precision = 0

            if len(true_terms) > 0:
                recall = len(true_terms & pred_terms) / len(true_terms)
            else:
                recall = 0

            precisions.append(precision)
            recalls.append(recall)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0

        f_scores.append(f1)

    return max(f_scores)
```

Higher is better.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| Top teams | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- ESM-2 protein language model
- Graph neural network on GO structure
- Ensemble of multiple models

### 2nd Place Solution

**Technique**:
- ProtTrans embeddings
- Attention-based aggregation
- Species-specific models

### 3rd Place Solution

**Innovation**:
- Multi-task learning across ontologies
- Sequence homology features
- Hierarchical loss function

---

## Common Winning Strategies

### 1. Protein Language Model Embeddings

```python
import torch
from transformers import EsmModel, EsmTokenizer

class ProteinEmbedder:
    def __init__(self, model_name='facebook/esm2_t33_650M_UR50D'):
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, sequence, pooling='mean'):
        """Get embedding for a protein sequence"""
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors='pt',
                               padding=True, truncation=True, max_length=1024)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state

        # Pool across sequence length
        if pooling == 'mean':
            # Exclude special tokens
            mask = inputs['attention_mask'].unsqueeze(-1)
            embedding = (hidden_states * mask).sum(1) / mask.sum(1)
        elif pooling == 'cls':
            embedding = hidden_states[:, 0, :]

        return embedding.numpy()
```

### 2. Hierarchical GO Prediction

```python
import networkx as nx
import numpy as np

class GOPredictor:
    def __init__(self, go_graph):
        self.go_graph = go_graph
        self.term_to_idx = {term: i for i, term in enumerate(go_graph.nodes())}

    def propagate_predictions(self, predictions):
        """
        Propagate predictions up the GO hierarchy
        If we predict term X, we should also predict all ancestors
        """
        propagated = predictions.copy()

        for term_idx, prob in enumerate(predictions):
            if prob > 0:
                term = list(self.term_to_idx.keys())[term_idx]
                # Get all ancestors
                ancestors = nx.ancestors(self.go_graph, term)
                for ancestor in ancestors:
                    if ancestor in self.term_to_idx:
                        anc_idx = self.term_to_idx[ancestor]
                        # Max probability with ancestor
                        propagated[anc_idx] = max(propagated[anc_idx], prob)

        return propagated

    def enforce_hierarchy(self, predictions):
        """
        Ensure predictions respect GO hierarchy
        Child probability <= parent probability
        """
        for term, prob in enumerate(predictions):
            term_name = list(self.term_to_idx.keys())[term]
            # Get parents
            parents = list(self.go_graph.predecessors(term_name))
            for parent in parents:
                if parent in self.term_to_idx:
                    parent_idx = self.term_to_idx[parent]
                    # Ensure parent >= child
                    if predictions[parent_idx] < prob:
                        predictions[term] = predictions[parent_idx]

        return predictions
```

### 3. Multi-Label Neural Network

```python
import torch
import torch.nn as nn

class GOClassifier(nn.Module):
    def __init__(self, embedding_dim=1280, num_terms=45000, hidden_dims=[512, 256]):
        super().__init__()

        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Separate heads for each ontology
        self.mf_head = nn.Linear(prev_dim, num_mf_terms)
        self.bp_head = nn.Linear(prev_dim, num_bp_terms)
        self.cc_head = nn.Linear(prev_dim, num_cc_terms)

    def forward(self, x):
        features = self.encoder(x)

        mf_logits = self.mf_head(features)
        bp_logits = self.bp_head(features)
        cc_logits = self.cc_head(features)

        return {
            'MF': torch.sigmoid(mf_logits),
            'BP': torch.sigmoid(bp_logits),
            'CC': torch.sigmoid(cc_logits)
        }

# Hierarchical loss
class HierarchicalBCELoss(nn.Module):
    def __init__(self, go_graph, term_to_idx):
        super().__init__()
        self.go_graph = go_graph
        self.term_to_idx = term_to_idx

    def forward(self, predictions, targets):
        # Standard BCE
        bce_loss = F.binary_cross_entropy(predictions, targets)

        # Hierarchy consistency loss
        hierarchy_loss = 0
        for term_idx in range(len(predictions[0])):
            term = list(self.term_to_idx.keys())[term_idx]
            parents = list(self.go_graph.predecessors(term))

            for parent in parents:
                if parent in self.term_to_idx:
                    parent_idx = self.term_to_idx[parent]
                    # Child should not exceed parent
                    violation = F.relu(predictions[:, term_idx] - predictions[:, parent_idx])
                    hierarchy_loss += violation.mean()

        return bce_loss + 0.1 * hierarchy_loss
```

### 4. Sequence Homology Features

```python
from Bio import pairwise2
from Bio.Blast import NCBIWWW, NCBIXML
import numpy as np

def get_homology_features(query_seq, database_proteins, database_annotations):
    """
    Transfer annotations from similar proteins
    """
    # Find similar proteins (simplified - real would use BLAST)
    similarities = []
    for db_seq in database_proteins:
        # Compute sequence similarity
        alignment = pairwise2.align.globalxx(query_seq[:100], db_seq[:100],
                                             score_only=True)
        similarity = alignment / min(len(query_seq), len(db_seq))
        similarities.append(similarity)

    # Weight annotations by similarity
    homology_scores = np.zeros(num_go_terms)

    for i, sim in enumerate(similarities):
        if sim > 0.3:  # Only use sufficiently similar
            for term in database_annotations[i]:
                homology_scores[term] += sim

    # Normalize
    if homology_scores.max() > 0:
        homology_scores /= homology_scores.max()

    return homology_scores
```

### 5. Ensemble and Calibration

```python
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def ensemble_predictions(models, X_test, weights=None):
    """Ensemble multiple protein function predictors"""
    if weights is None:
        weights = [1/len(models)] * len(models)

    all_predictions = []
    for model, weight in zip(models, weights):
        preds = model.predict(X_test)
        all_predictions.append(preds * weight)

    ensemble_preds = np.sum(all_predictions, axis=0)

    return ensemble_preds

def calibrate_predictions(predictions, y_val):
    """Calibrate predictions for each GO term"""
    calibrated = np.zeros_like(predictions)

    for term_idx in range(predictions.shape[1]):
        # Find optimal threshold for this term
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (predictions[:, term_idx] >= threshold).astype(int)
            f1 = f1_score(y_val[:, term_idx], y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Apply calibration
        calibrated[:, term_idx] = predictions[:, term_idx] / best_threshold * 0.5

    return np.clip(calibrated, 0, 1)
```

---

## Technical Insights

### Protein Language Models

| Model | Parameters | Description |
|-------|------------|-------------|
| ESM-2 | 650M-15B | Meta's protein LM |
| ProtTrans | 3-11B | Multiple architectures |
| UniRep | 64M | Earlier model |

### GO Term Statistics

| Ontology | Terms | Avg Annotations |
|----------|-------|-----------------|
| MF | ~12,000 | 3-5 per protein |
| BP | ~30,000 | 5-10 per protein |
| CC | ~4,000 | 2-4 per protein |

### Key Challenges

| Challenge | Approach |
|-----------|----------|
| Hierarchy | Propagation, hierarchical loss |
| Imbalance | Weighted loss, threshold tuning |
| Long sequences | Truncation, chunking |
| Sparse labels | Transfer learning |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | 9 hours |
| GPU Runtime | 9 hours |
| Internet | Disabled |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 2nd | [Discussion](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/434064) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/464437) |
| 4th | [Discussion](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/433732) |
| 5th | [Discussion](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/463009) |

---

## Citation

```bibtex
@misc{cafa-5-protein-function-prediction,
    author = {CAFA},
    title = {CAFA 5 Protein Function Prediction},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/cafa-5-protein-function-prediction}},
    note = {Kaggle}
}
```
