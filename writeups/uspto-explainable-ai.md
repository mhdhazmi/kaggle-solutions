# USPTO - Explainable AI for Patent Professionals

> Help patent professionals understand AI results through a familiar query language

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Text Generation / Query Synthesis |
| **Data Domain** | Legal / Patents / Search |
| **ML Approach** | LLM Fine-tuning + Rule-based Systems |
| **Key Techniques** | Query Language Generation, Information Retrieval |
| **Difficulty** | Advanced |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Featured Competition |
| **Total Prize** | $50,000 |
| **Teams** | 571 |
| **Timeline** | 2024 |
| **Evaluation Metric** | Custom USPTO Metric |
| **Host** | USPTO (United States Patent and Trademark Office) |

## Problem Description

Generate human-readable patent search queries that explain why an AI system retrieved certain patent documents. Bridge the gap between black-box AI retrieval and the structured query language patent professionals already use.

### The Challenge

- Given: AI-retrieved patent documents for a query
- Output: Explainable search query in patent search language
- Goal: Help professionals understand and verify AI decisions
- Constraint: Queries must be valid in patent search systems

### Why It Matters

- **Explainability**: AI decisions need human verification
- **Trust**: Patent examiners need to understand search results
- **Efficiency**: Structured queries enable refinement
- **Legal Standards**: Patent examination requires explainable reasoning

---

## Data Description

### Patent Search Context

Patent search involves:
- Boolean operators (AND, OR, NOT)
- Field-specific searches (title, abstract, claims)
- Classification codes (CPC, IPC)
- Date ranges and applicant searches

### Query Language

Patent professionals use structured queries like:
```
((artificial AND intelligence) OR (machine AND learning))
AND (medical OR healthcare)
AND CPC=(G06N OR G16H)
```

### Task Format

| Input | Output |
|-------|--------|
| Retrieved patent documents | Explainable search query |
| AI relevance scores | Query that retrieves similar documents |
| Patent metadata | Valid patent search syntax |

---

## Evaluation

**Metric**: Custom USPTO Metric

Evaluates:
1. Query syntactic validity
2. Retrieval overlap with original results
3. Query interpretability
4. Precision/recall of retrieved documents

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st-5th | Share of $50,000 |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- LLM fine-tuning on patent queries
- Rule-based query validation
- Iterative refinement
- CPC code extraction

### 2nd Place Solution

**Technique**:
- Hybrid neural-symbolic approach
- Template-based generation
- Query optimization

### 3rd Place Solution

**Innovation**:
- Few-shot learning with examples
- Query simplification
- Error correction

---

## Common Winning Strategies

### 1. Understanding Patent Query Syntax

```python
# Patent search query components
query_components = {
    'boolean': ['AND', 'OR', 'NOT', 'ANDNOT'],
    'fields': ['TTL/', 'ABST/', 'CLMS/', 'SPEC/', 'ACLM/'],
    'cpc': 'CPC/',  # Cooperative Patent Classification
    'dates': ['APD/', 'ISD/'],  # Application date, Issue date
    'applicant': 'AN/',
    'inventor': 'IN/',
}

# Example valid query
valid_query = """
TTL/(machine learning) AND
ABST/(neural network) AND
CPC/(G06N3/08) AND
APD/20200101->20231231
"""
```

### 2. Query Generation from Documents

```python
def generate_query_from_patents(patents):
    """Generate explainable query from retrieved patents"""

    # Extract common terms
    common_terms = extract_common_terms(patents)

    # Extract CPC codes
    cpc_codes = extract_cpc_codes(patents)

    # Build query
    query_parts = []

    # Add term-based search
    if common_terms:
        term_query = ' AND '.join(f'({term})' for term in common_terms[:5])
        query_parts.append(f'ABST/({term_query})')

    # Add CPC codes
    if cpc_codes:
        cpc_query = ' OR '.join(cpc_codes[:3])
        query_parts.append(f'CPC/({cpc_query})')

    return ' AND '.join(query_parts)
```

### 3. LLM-based Query Generation

```python
query_generation_prompt = """
You are a patent search expert. Given a set of patent documents retrieved by an AI system,
generate an explainable search query in USPTO patent search syntax.

Retrieved Patents:
{patent_summaries}

Requirements:
1. Use valid USPTO query syntax
2. Include relevant CPC classification codes
3. Use Boolean operators (AND, OR, NOT)
4. Focus on key technical terms
5. Query should retrieve similar patents

Generate a patent search query:
"""

def generate_with_llm(patents):
    summaries = format_patent_summaries(patents)
    prompt = query_generation_prompt.format(patent_summaries=summaries)
    query = llm.generate(prompt)
    return validate_and_clean(query)
```

### 4. Query Validation and Correction

```python
def validate_query(query):
    """Validate patent search query syntax"""
    errors = []

    # Check balanced parentheses
    if query.count('(') != query.count(')'):
        errors.append('Unbalanced parentheses')

    # Check valid operators
    valid_ops = ['AND', 'OR', 'NOT', 'ANDNOT', 'NEAR', 'ADJ']
    for word in query.split():
        if word.isupper() and word not in valid_ops:
            if not any(word.startswith(f) for f in ['CPC/', 'TTL/', 'ABST/']):
                errors.append(f'Invalid operator: {word}')

    # Check CPC format
    cpc_pattern = r'CPC/\([A-Z][0-9]{2}[A-Z][0-9/]+\)'
    # ... additional validation

    return len(errors) == 0, errors

def correct_query(query):
    """Attempt to fix common query errors"""
    # Balance parentheses
    while query.count('(') > query.count(')'):
        query += ')'
    while query.count('(') < query.count(')'):
        query = '(' + query

    # Fix spacing around operators
    for op in ['AND', 'OR', 'NOT']:
        query = re.sub(f'\\s*{op}\\s*', f' {op} ', query)

    return query
```

### 5. CPC Code Extraction

```python
def extract_relevant_cpc(patents):
    """Extract most relevant CPC codes from patent set"""

    # Count CPC code occurrences
    cpc_counts = Counter()
    for patent in patents:
        for cpc in patent['cpc_codes']:
            # Use different granularities
            cpc_counts[cpc[:4]] += 1  # Subclass
            cpc_counts[cpc[:7]] += 1  # Group

    # Return most common
    return [cpc for cpc, count in cpc_counts.most_common(5)]

# CPC hierarchy example
# G06N - Computing arrangements based on specific computational models
# G06N3 - Computing arrangements based on biological models
# G06N3/08 - Learning methods
```

---

## Technical Insights

### Patent Search Systems

| System | Query Language |
|--------|----------------|
| USPTO PatFT/AppFT | USPTO syntax |
| Google Patents | Natural language + filters |
| Espacenet | European syntax |
| WIPO | PCT syntax |

### Query Complexity Trade-offs

| Query Type | Precision | Recall | Explainability |
|------------|-----------|--------|----------------|
| Simple terms | Low | High | High |
| Complex Boolean | High | Medium | Medium |
| CPC-heavy | Variable | Focused | Domain expert |

### Explainability Requirements

For patent examination:
1. **Reproducibility**: Others can run the same search
2. **Transparency**: Clear why documents were retrieved
3. **Refinability**: Can modify to narrow/broaden
4. **Documentation**: Supports legal requirements

### Common Query Patterns

```
# Technology-focused
TTL/(term1 OR term2) AND ABST/(term3 AND term4)

# Classification-focused
CPC/(A61B5/00) AND ABST/(monitoring)

# Applicant/date-focused
AN/(company) AND APD/2020->2024

# Proximity search
CLMS/(neural ADJ network)  # Adjacent words
```

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/c/uspto-explainable-ai/discussion/522233) |
| 2nd | [Discussion](https://www.kaggle.com/c/uspto-explainable-ai/discussion/522258) |
| 3rd | [Discussion](https://www.kaggle.com/c/uspto-explainable-ai/discussion/522639) |
| 4th | [Discussion](https://www.kaggle.com/c/uspto-explainable-ai/discussion/522200) |
| 5th | [Discussion](https://www.kaggle.com/c/uspto-explainable-ai/discussion/522201) |

---

## Resources

- [USPTO Patent Full-Text Database](https://patft.uspto.gov/)
- [CPC Classification](https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html)

---

## Citation

```bibtex
@misc{uspto-explainable-ai,
    author = {USPTO},
    title = {USPTO - Explainable AI for Patent Professionals},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/uspto-explainable-ai}},
    note = {Kaggle}
}
```
