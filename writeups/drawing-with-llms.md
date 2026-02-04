# Drawing with LLMs

> Build and submit Kaggle Packages capable of generating SVG images of specific concepts

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Generative AI / Text-to-SVG |
| **Data Domain** | Computer Graphics / Vector Art |
| **ML Approach** | Large Language Models |
| **Key Techniques** | Prompt Engineering, SVG Generation, Fine-tuning |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Code Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,309 |
| **Timeline** | 2025 |
| **Evaluation Metric** | SVG Image Fidelity |
| **Host** | Kaggle |

## Problem Description

Create AI systems that can generate SVG (Scalable Vector Graphics) images from text descriptions of concepts.

### The Challenge

- Take a text prompt describing a concept
- Generate a valid SVG that visually represents the concept
- SVG must render correctly and match the description
- Work within SVG format constraints

### Why SVGs?

- **Scalable**: Resolution-independent graphics
- **Compact**: Text-based, efficient representation
- **Structured**: Programmatic, verifiable output
- **Challenging**: Requires understanding of both language and visual composition

---

## Data Description

### Task Format

| Input | Output |
|-------|--------|
| Text concept/prompt | Valid SVG code |

### SVG Basics

SVG elements include:
- `<rect>`: Rectangles
- `<circle>`: Circles
- `<ellipse>`: Ellipses
- `<line>`: Lines
- `<polyline>`: Connected lines
- `<polygon>`: Closed shapes
- `<path>`: Complex curves (Bezier paths)
- `<text>`: Text elements

### Example

Prompt: "A red circle"
```xml
<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" fill="red"/>
</svg>
```

---

## Evaluation

**Metric**: SVG Image Fidelity

Evaluates how well the generated SVG matches the intended concept:
- Visual similarity to expected representation
- Correct SVG syntax
- Appropriate use of SVG elements

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
- Fine-tuned LLM on SVG generation task
- Structured output parsing
- Iterative refinement of generated SVGs

### 2nd Place Solution

**Technique**:
- Multi-stage generation pipeline
- Template-based SVG construction
- LLM for element placement and styling

### 3rd Place Solution

**Innovation**:
- Code-specialized LLM
- Post-processing for SVG validity
- Ensemble of generation strategies

### 4th Place Solution

**Approach**:
- Prompt engineering optimization
- Few-shot learning with SVG examples
- Constraint-guided generation

---

## Common Winning Strategies

### 1. LLM Selection

| Model Type | Strengths |
|------------|-----------|
| Code LLMs | Better at structured output |
| General LLMs | Better concept understanding |
| Fine-tuned | Task-specific optimization |

### 2. Prompt Engineering

Effective prompts include:
- Clear SVG structure instructions
- Element specification
- Color and positioning guidelines
- Examples (few-shot)

```
Generate an SVG image of [concept].
Use basic SVG elements.
The SVG should be 200x200 pixels.
Example format:
<svg width="200" height="200">
  [elements here]
</svg>
```

### 3. Output Validation

```python
def validate_svg(svg_string):
    # Check XML validity
    # Verify SVG namespace
    # Ensure renderable elements
    # Check for common errors
    pass
```

### 4. Iterative Refinement

```
Initial Generation → Validation → Feedback → Refinement → Final SVG
```

### 5. Template Approaches

Pre-defined templates for common concepts:
- Shapes: circles, squares, triangles
- Objects: house, tree, car
- Abstract: patterns, gradients

---

## Technical Insights

### SVG Generation Challenges

1. **Coordinate system**: LLMs struggle with precise positioning
2. **Path commands**: Complex Bezier curves are difficult
3. **Color naming**: Inconsistent color representation
4. **Proportions**: Maintaining visual balance
5. **Complexity**: Balancing detail with validity

### SVG Best Practices

```xml
<!-- Use viewBox for scalability -->
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">

  <!-- Group related elements -->
  <g fill="blue">
    <circle cx="25" cy="25" r="10"/>
    <circle cx="75" cy="25" r="10"/>
  </g>

  <!-- Use meaningful colors -->
  <rect x="10" y="60" width="80" height="20" fill="#FF5733"/>

</svg>
```

### Common Errors

| Error | Solution |
|-------|----------|
| Invalid XML | XML parser validation |
| Missing namespace | Add `xmlns` attribute |
| Unclosed tags | Auto-close or reject |
| Invalid attributes | Whitelist valid attributes |

### LLM Output Parsing

```python
import re

def extract_svg(llm_output):
    # Find SVG content
    svg_match = re.search(r'<svg.*?</svg>', llm_output, re.DOTALL)
    if svg_match:
        return svg_match.group(0)
    return None
```

---

## Kaggle Packages

This competition uses Kaggle Packages for submission:
- Containerized code execution
- Reproducible environment
- Offline inference

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/drawing-with-llms/discussion/581027) |
| 2nd | [Discussion](https://www.kaggle.com/c/drawing-with-llms/discussion/581023) |
| 3rd | [Discussion](https://www.kaggle.com/c/drawing-with-llms/discussion/581024) |
| 4th | [Discussion](https://www.kaggle.com/c/drawing-with-llms/discussion/581108) |
| 5th | [Discussion](https://www.kaggle.com/c/drawing-with-llms/discussion/581128) |

---

## Citation

```bibtex
@misc{drawing-with-llms,
    title = {Drawing with LLMs},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/drawing-with-llms}},
    note = {Kaggle}
}
```
