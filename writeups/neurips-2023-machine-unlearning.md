# NeurIPS 2023 - Machine Unlearning

> Erase the influence of requested samples without hurting accuracy

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Model Modification / Privacy |
| **Data Domain** | Machine Learning / Privacy |
| **ML Approach** | Fine-tuning, Influence Functions |
| **Key Techniques** | Selective Forgetting, Model Surgery, Knowledge Distillation |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,188 |
| **Timeline** | July - November 2023 |
| **Evaluation Metric** | Custom Unlearning Metric |
| **Host** | NeurIPS 2023 |

## Problem Description

"Unlearn" specific training samples from a trained model - make the model behave as if those samples were never in the training data.

### The Challenge

- Remove influence of "forget set" samples
- Maintain accuracy on "retain set"
- Be indistinguishable from retraining from scratch
- Do it efficiently (faster than retraining)

### Why It Matters

- **Privacy (GDPR)**: Right to be forgotten
- **Copyright**: Remove copyrighted training data
- **Bias Correction**: Remove biased samples
- **Security**: Prevent data poisoning attacks

---

## Data Description

### Setup

A pre-trained image classification model (ResNet-18 on modified CIFAR-10) with:
- Original training set
- Forget set (samples to unlearn)
- Retain set (samples to keep)

### Files

| File | Description |
|------|-------------|
| `original_model.pth` | Pre-trained model weights |
| `retain_loader` | DataLoader for retain samples |
| `forget_loader` | DataLoader for forget samples |
| `test_loader` | Test set for evaluation |

### Task

Transform the original model so that:
1. It performs well on retain and test sets
2. The forget set has no detectable influence
3. Statistically similar to model trained without forget set

---

## Evaluation

**Metric**: Custom Unlearning Score

```python
def unlearning_metric(unlearned_model, original_model, retrained_model,
                      forget_loader, retain_loader, test_loader):
    """
    Evaluate unlearning quality

    Combines:
    1. Utility: accuracy on retain/test sets
    2. Forgetting: behavior on forget set matches retrained
    3. Efficiency: time compared to retraining
    """
    # Utility score
    retain_acc = evaluate_accuracy(unlearned_model, retain_loader)
    test_acc = evaluate_accuracy(unlearned_model, test_loader)

    # Forgetting score (statistical distance from retrained)
    forget_outputs_unlearned = get_outputs(unlearned_model, forget_loader)
    forget_outputs_retrained = get_outputs(retrained_model, forget_loader)
    forget_distance = compute_distribution_distance(
        forget_outputs_unlearned,
        forget_outputs_retrained
    )

    # Membership inference attack resistance
    mia_auc = membership_inference_attack(unlearned_model, forget_loader)

    # Combine scores
    score = combine_scores(retain_acc, test_acc, forget_distance, mia_auc)

    return score
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
- Gradient ascent on forget set
- Fine-tuning on retain set
- Careful balancing of objectives

### 2nd Place Solution

**Technique**:
- Knowledge distillation from modified teacher
- Selective layer fine-tuning
- Output calibration

### 3rd Place Solution

**Innovation**:
- Fisher information-based approach
- Influence function approximation
- Efficient parameter updates

---

## Common Winning Strategies

### 1. Gradient Ascent on Forget Set

```python
import torch
import torch.nn.functional as F

def unlearn_gradient_ascent(model, forget_loader, epochs=5, lr=0.001):
    """
    Make model worse on forget set by gradient ascent
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, labels in forget_loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)

            # Negative loss = gradient ascent
            loss = -F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()

    return model
```

### 2. Fine-Tune on Retain Set

```python
def finetune_retain(model, retain_loader, epochs=10, lr=0.0001):
    """
    Fine-tune on retain set to restore performance
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in retain_loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()

    return model
```

### 3. Combined Unlearning Approach

```python
def combined_unlearning(model, forget_loader, retain_loader,
                        alpha=0.1, epochs=10, lr=0.001):
    """
    Combined loss: maximize on forget, minimize on retain
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    forget_iter = iter(forget_loader)

    for epoch in range(epochs):
        for retain_images, retain_labels in retain_loader:
            retain_images = retain_images.cuda()
            retain_labels = retain_labels.cuda()

            # Get forget batch
            try:
                forget_images, forget_labels = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_images, forget_labels = next(forget_iter)

            forget_images = forget_images.cuda()
            forget_labels = forget_labels.cuda()

            optimizer.zero_grad()

            # Retain loss (minimize)
            retain_outputs = model(retain_images)
            retain_loss = F.cross_entropy(retain_outputs, retain_labels)

            # Forget loss (maximize = negative minimize)
            forget_outputs = model(forget_images)
            forget_loss = -F.cross_entropy(forget_outputs, forget_labels)

            # Combined loss
            loss = retain_loss + alpha * forget_loss

            loss.backward()
            optimizer.step()

    return model
```

### 4. Knowledge Distillation Approach

```python
def unlearn_with_distillation(student_model, teacher_model,
                              forget_set_ids, retain_loader, temperature=4):
    """
    Distill knowledge but exclude forget set patterns
    """
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    teacher_model.eval()
    student_model.train()

    for images, labels, sample_ids in retain_loader:
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        # Get student predictions
        student_outputs = student_model(images)

        # Soft targets (distillation)
        soft_loss = F.kl_div(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # Hard targets
        hard_loss = F.cross_entropy(student_outputs, labels)

        # Combined
        loss = 0.5 * soft_loss + 0.5 * hard_loss

        loss.backward()
        optimizer.step()

    return student_model
```

### 5. Fisher Information-Based Unlearning

```python
def compute_fisher_information(model, data_loader):
    """
    Compute Fisher Information Matrix diagonal approximation
    """
    fisher = {name: torch.zeros_like(param)
              for name, param in model.named_parameters()}

    model.eval()
    for images, labels in data_loader:
        images, labels = images.cuda(), labels.cuda()

        model.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

    # Normalize
    for name in fisher:
        fisher[name] /= len(data_loader.dataset)

    return fisher

def unlearn_fisher(model, forget_loader, retain_fisher, lambda_reg=0.1):
    """
    Unlearn by moving parameters in directions
    with low Fisher information for retain set
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for images, labels in forget_loader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)

        # Forget loss (maximize)
        forget_loss = -F.cross_entropy(outputs, labels)

        # Regularize: prefer directions with low retain Fisher
        reg_loss = 0
        for name, param in model.named_parameters():
            if name in retain_fisher:
                reg_loss += torch.sum(retain_fisher[name] * (param ** 2))

        loss = forget_loss + lambda_reg * reg_loss

        loss.backward()
        optimizer.step()

    return model
```

---

## Technical Insights

### Unlearning Approaches

| Approach | Pros | Cons |
|----------|------|------|
| Gradient ascent | Simple | May hurt retain performance |
| Fine-tuning | Restores performance | May not fully forget |
| Distillation | Stable | Requires clean teacher |
| Fisher-based | Principled | Computationally expensive |

### Evaluation Criteria

| Criterion | Description |
|-----------|-------------|
| Utility | Performance on retain/test |
| Forgetting | No trace of forget samples |
| MIA resistance | Resist membership inference |
| Efficiency | Faster than retraining |

### Key Challenges

| Challenge | Solution |
|-----------|----------|
| Catastrophic forgetting of retain | Careful learning rate, regularization |
| Incomplete forgetting | Multiple rounds, stronger gradient ascent |
| Efficiency | Layer-wise updates, approximate methods |

---

## Code Competition Requirements

| Requirement | Limit |
|-------------|-------|
| Runtime | Limited |
| Internet | Disabled |
| Memory | Standard |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 2nd | [Discussion](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/458721) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/459200) |
| 4th | [Discussion](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/459334) |
| 5th | [Discussion](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/459148) |

---

## Citation

```bibtex
@misc{neurips-2023-machine-unlearning,
    author = {NeurIPS},
    title = {NeurIPS 2023 - Machine Unlearning},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/neurips-2023-machine-unlearning}},
    note = {Kaggle}
}
```
