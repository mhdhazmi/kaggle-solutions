# Google - Fast or Slow? Predict AI Model Runtime

> Predict how fast an AI model runs

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Regression on Graph Data |
| **Data Domain** | ML Systems / Compiler Optimization |
| **ML Approach** | Graph Neural Networks |
| **Key Techniques** | Graph Learning, Node Embeddings, Attention |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 616 |
| **Timeline** | August - November 2023 |
| **Evaluation Metric** | Custom TPU Runtime Metric |
| **Host** | Google |

## Problem Description

Predict the runtime of machine learning models on Google TPUs based on their computational graph structure - enabling better compiler optimization.

### The Challenge

- Predict runtime from computational graph representation
- Handle variable-size graphs
- Generalize to unseen model architectures
- Multiple prediction tracks (layout, tile)

### Why It Matters

- **Compiler Optimization**: Better layout/tiling decisions
- **Resource Allocation**: Predict compute needs
- **Model Selection**: Compare models before training
- **Hardware Design**: Understand performance bottlenecks

---

## Data Description

### TPU Graph Representation

ML models represented as computational graphs:
- **Nodes**: Operations (MatMul, Conv, etc.)
- **Edges**: Data dependencies
- **Features**: Operation attributes

### Files

| File | Description |
|------|-------------|
| `npz/` | Graph data in NPZ format |
| `*_runtime.csv` | Runtime measurements |
| `sample_submission.csv` | Submission format |

### Graph Features

| Feature Type | Description |
|--------------|-------------|
| Node features | Operation type, dimensions, configs |
| Edge features | Data flow, tensor shapes |
| Graph features | Model architecture info |
| Config features | Layout/tiling configuration |

### Tracks

| Track | Description |
|-------|-------------|
| Layout | Predict runtime for different data layouts |
| Tile | Predict runtime for different tiling configs |

---

## Evaluation

**Metric**: Slowdown with respect to best configuration

```python
def compute_slowdown(predictions, runtimes, configs):
    """
    For each graph, measure how far from optimal config
    """
    slowdowns = []

    for graph_id in predictions.keys():
        # Get predicted best config
        pred_best_config = np.argmin(predictions[graph_id])

        # Get actual runtimes
        actual_runtimes = runtimes[graph_id]

        # Actual best config
        actual_best_config = np.argmin(actual_runtimes)

        # Slowdown = runtime of predicted / runtime of best
        slowdown = actual_runtimes[pred_best_config] / actual_runtimes[actual_best_config]
        slowdowns.append(slowdown)

    return np.mean(slowdowns)
```

Lower slowdown is better (1.0 = perfect).

---

## Prize Structure

| Place | Prize |
|-------|-------|
| Top teams | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Graph Transformer architecture
- Multi-scale node features
- Ensemble of different architectures

### 2nd Place Solution

**Technique**:
- Message passing neural network
- Attention-based aggregation
- Feature engineering on op types

### 3rd Place Solution

**Innovation**:
- Hierarchical graph representation
- Config embedding network
- Transfer learning across tracks

---

## Common Winning Strategies

### 1. Graph Data Loading

```python
import numpy as np
import torch
from torch_geometric.data import Data

def load_graph(npz_path):
    """Load TPU graph from NPZ file"""
    data = np.load(npz_path)

    # Node features
    node_feat = torch.tensor(data['node_feat'], dtype=torch.float32)
    node_opcode = torch.tensor(data['node_opcode'], dtype=torch.long)

    # Edge information
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    # Config features (for layout/tile prediction)
    config_feat = torch.tensor(data['config_feat'], dtype=torch.float32)

    # Create PyG Data object
    graph = Data(
        x=node_feat,
        edge_index=edge_index,
        node_opcode=node_opcode,
        config_feat=config_feat
    )

    return graph

def create_node_features(node_feat, node_opcode, num_opcodes=120):
    """Combine numeric and categorical node features"""
    # One-hot encode opcodes
    opcode_onehot = torch.zeros(len(node_opcode), num_opcodes)
    opcode_onehot.scatter_(1, node_opcode.unsqueeze(1), 1)

    # Concatenate
    combined = torch.cat([node_feat, opcode_onehot], dim=1)
    return combined
```

### 2. Graph Neural Network Model

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class TPURuntimePredictor(nn.Module):
    def __init__(self, node_feat_dim, config_feat_dim, hidden_dim=256, num_heads=4):
        super().__init__()

        # Node feature encoder
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        # Graph attention layers
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)

        # Config encoder
        self.config_encoder = nn.Sequential(
            nn.Linear(config_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        config_feat = data.config_feat
        batch = data.batch

        # Encode node features
        x = F.relu(self.node_encoder(x))

        # Graph attention layers
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = F.relu(self.gat3(x, edge_index))

        # Global pooling
        graph_emb = global_mean_pool(x, batch)

        # Encode config
        config_emb = self.config_encoder(config_feat)

        # Combine graph and config
        combined = torch.cat([graph_emb, config_emb], dim=1)

        # Predict runtime
        runtime = self.predictor(combined)

        return runtime.squeeze(-1)
```

### 3. Message Passing with Edge Features

```python
from torch_geometric.nn import MessagePassing

class TPUConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='add')

        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_update = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Node transformation
        x = self.lin_node(x)

        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update
        out = self.lin_update(torch.cat([x, out], dim=1))

        return F.relu(out)

    def message(self, x_j, edge_attr):
        # Message from neighbor + edge features
        edge_emb = self.lin_edge(edge_attr)
        return x_j + edge_emb

class TPUGraphNet(nn.Module):
    def __init__(self, node_dim, edge_dim, config_dim, hidden_dim=256):
        super().__init__()

        self.conv1 = TPUConv(node_dim, hidden_dim, edge_dim)
        self.conv2 = TPUConv(hidden_dim, hidden_dim, edge_dim)
        self.conv3 = TPUConv(hidden_dim, hidden_dim, edge_dim)

        self.config_mlp = nn.Sequential(
            nn.Linear(config_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output = nn.Linear(hidden_dim * 2, 1)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = self.conv3(x, data.edge_index, data.edge_attr)

        # Pool graph
        graph_emb = global_mean_pool(x, data.batch)

        # Config embedding
        config_emb = self.config_mlp(data.config_feat)

        # Predict
        combined = torch.cat([graph_emb, config_emb], dim=1)
        return self.output(combined).squeeze(-1)
```

### 4. Ranking Loss for Config Selection

```python
def pairwise_ranking_loss(predictions, runtimes, margin=0.1):
    """
    If runtime_i < runtime_j, pred_i should be < pred_j
    """
    n = len(predictions)
    loss = 0
    count = 0

    for i in range(n):
        for j in range(n):
            if runtimes[i] < runtimes[j]:
                # pred_i should be smaller
                diff = predictions[i] - predictions[j] + margin
                loss += F.relu(diff)
                count += 1

    return loss / (count + 1)

def listwise_loss(predictions, runtimes):
    """
    Softmax ranking loss
    """
    # Convert to ranking probabilities
    pred_probs = F.softmax(-predictions, dim=0)  # Lower pred = higher prob
    true_probs = F.softmax(-runtimes, dim=0)     # Lower runtime = higher prob

    # KL divergence
    return F.kl_div(pred_probs.log(), true_probs, reduction='batchmean')
```

### 5. Feature Engineering

```python
def engineer_graph_features(graph_data):
    """Extract additional graph-level features"""
    features = {}

    # Graph structure features
    features['num_nodes'] = graph_data.num_nodes
    features['num_edges'] = graph_data.num_edges
    features['density'] = graph_data.num_edges / (graph_data.num_nodes ** 2)

    # Node type statistics
    opcodes = graph_data.node_opcode.numpy()
    unique, counts = np.unique(opcodes, return_counts=True)
    for opcode, count in zip(unique, counts):
        features[f'opcode_{opcode}_count'] = count

    # Degree statistics
    edge_index = graph_data.edge_index.numpy()
    in_degrees = np.bincount(edge_index[1], minlength=graph_data.num_nodes)
    out_degrees = np.bincount(edge_index[0], minlength=graph_data.num_nodes)

    features['avg_in_degree'] = in_degrees.mean()
    features['max_in_degree'] = in_degrees.max()
    features['avg_out_degree'] = out_degrees.mean()

    # Critical path estimate (longest path)
    # features['critical_path'] = compute_critical_path(graph_data)

    return features
```

---

## Technical Insights

### TPU Compilation

| Concept | Description |
|---------|-------------|
| Layout | How tensors are stored in memory |
| Tiling | How operations are split across cores |
| Fusion | Combining operations |
| XLA | Compiler framework |

### Graph Structure Patterns

| Pattern | Impact on Runtime |
|---------|-------------------|
| Sequential ops | Memory bound |
| Wide parallelism | Compute bound |
| Reduction ops | Communication |
| Large tensors | Memory bandwidth |

### Model Architecture Effects

| Architecture | Typical Characteristics |
|--------------|------------------------|
| Transformers | Attention ops, large matrices |
| CNNs | Conv ops, spatial locality |
| RNNs | Sequential dependencies |
| MLPs | Dense operations |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| CPU Runtime | Variable |
| GPU Runtime | Variable |
| Internet | Disabled |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456343) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456365) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456377) |
| 4th | [Discussion](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456462) |
| 5th | [Discussion](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456093) |

---

## Citation

```bibtex
@misc{predict-ai-model-runtime,
    author = {Google},
    title = {Google - Fast or Slow? Predict AI Model Runtime},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/predict-ai-model-runtime}},
    note = {Kaggle}
}
```
