# Santa 2023 - The Polytope Permutation Puzzle

> Solve a complex combinatorial optimization puzzle

## Competition Classification

| Category | Value |
|----------|-------|
| **Problem Type** | Combinatorial Optimization |
| **Data Domain** | Mathematics / Puzzles |
| **ML Approach** | Search Algorithms + Heuristics |
| **Key Techniques** | Beam Search, Local Search, Group Theory |
| **Difficulty** | Expert |

## Competition Overview

| Attribute | Value |
|-----------|-------|
| **Competition Type** | Research Optimization Competition |
| **Total Prize** | $50,000 |
| **Teams** | 1,054 |
| **Timeline** | December 2023 - January 2024 |
| **Evaluation Metric** | Total Moves (Lower is Better) |
| **Host** | Kaggle |

## Problem Description

Solve a collection of permutation puzzles based on polytope symmetries - rotating elements to reach a goal configuration in minimum moves.

### The Challenge

- Solve 398 different permutation puzzles
- Minimize total number of moves across all puzzles
- Puzzles based on mathematical polytopes (cube, globe, wreath)
- Each puzzle type has different allowed moves

### Puzzle Types

| Type | Description | Complexity |
|------|-------------|------------|
| Cube | 3D Rubik's cube variants | High |
| Globe | Spherical rotation puzzle | Medium-High |
| Wreath | Circular permutation rings | Medium |

### Why It Matters

- **Algorithmic Research**: Tests search algorithm efficiency
- **Group Theory**: Applications of mathematical symmetry
- **AI Planning**: Sequential decision making
- **Puzzle Solving**: Automated theorem proving connections

---

## Data Description

### Files

| File | Description |
|------|-------------|
| `puzzles.csv` | Puzzle definitions with initial/goal states |
| `puzzle_info.csv` | Puzzle type information |
| `sample_submission.csv` | Submission format |

### Puzzle Format

| Column | Description |
|--------|-------------|
| `id` | Puzzle identifier |
| `puzzle_type` | Type (cube_x/x/x, globe_x/x, wreath_x/x) |
| `solution_state` | Goal configuration (semicolon-separated) |
| `initial_state` | Starting configuration |
| `num_wildcards` | Allowed "don't care" positions |

### Move Notation

Moves are named based on the puzzle type:
- Cube: Face rotations (f, r, u, d, l, b) and their variants
- Globe: Rotation around axes
- Wreath: Ring rotations

---

## Evaluation

**Metric**: Total number of moves across all puzzles

```python
def evaluate(submission_df, puzzles_df):
    """
    Score = Sum of move counts for all puzzles
    Lower is better
    """
    total_moves = 0

    for _, row in submission_df.iterrows():
        puzzle_id = row['id']
        moves = row['moves'].split('.')

        # Verify solution is valid
        if verify_solution(puzzle_id, moves, puzzles_df):
            total_moves += len(moves)
        else:
            total_moves += PENALTY  # Large penalty for invalid

    return total_moves
```

Lower is better. Invalid solutions receive heavy penalties.

---

## Prize Structure

| Place | Prize |
|-------|-------|
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $6,000 |
| 5th | $5,000 |
| 6th-10th | $1,200 each |

---

## Top Solutions

### 1st Place Solution

**Key Approach**:
- Kociemba algorithm for cube puzzles
- Custom solvers per puzzle type
- Iterative deepening search
- Move sequence optimization

### 2nd Place Solution

**Technique**:
- A* search with admissible heuristics
- Pattern databases for cubes
- Genetic algorithms for wreath puzzles

### 3rd Place Solution

**Innovation**:
- Machine learning for move selection
- Monte Carlo Tree Search
- Subgroup-based solving

---

## Common Winning Strategies

### 1. Generic Permutation Puzzle Framework

```python
import numpy as np
from collections import deque

class PermutationPuzzle:
    def __init__(self, puzzle_type, initial_state, goal_state, moves_dict):
        self.puzzle_type = puzzle_type
        self.initial = np.array(initial_state)
        self.goal = np.array(goal_state)
        self.moves = moves_dict  # {move_name: permutation}
        self.inverse_moves = self._compute_inverses()

    def _compute_inverses(self):
        """Compute inverse for each move"""
        inverses = {}
        for name, perm in self.moves.items():
            # Inverse permutation
            inv = np.zeros_like(perm)
            for i, p in enumerate(perm):
                inv[p] = i
            if name.endswith("'"):
                inverses[name] = name[:-1]
            else:
                inverses[name] = name + "'"
        return inverses

    def apply_move(self, state, move_name):
        """Apply a move to a state"""
        perm = self.moves[move_name]
        return state[perm]

    def apply_sequence(self, state, move_sequence):
        """Apply a sequence of moves"""
        for move in move_sequence:
            state = self.apply_move(state, move)
        return state

    def is_solved(self, state, wildcards=None):
        """Check if state matches goal"""
        if wildcards is None:
            return np.array_equal(state, self.goal)
        else:
            # Ignore wildcard positions
            mask = ~np.isin(range(len(state)), wildcards)
            return np.array_equal(state[mask], self.goal[mask])
```

### 2. Breadth-First Search (BFS)

```python
from collections import deque

def bfs_solve(puzzle, max_depth=20):
    """BFS solver for small puzzles"""
    initial_tuple = tuple(puzzle.initial)
    goal_tuple = tuple(puzzle.goal)

    if initial_tuple == goal_tuple:
        return []

    queue = deque([(initial_tuple, [])])
    visited = {initial_tuple}

    while queue:
        state, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for move_name in puzzle.moves:
            new_state = puzzle.apply_move(np.array(state), move_name)
            new_tuple = tuple(new_state)

            if new_tuple == goal_tuple:
                return path + [move_name]

            if new_tuple not in visited:
                visited.add(new_tuple)
                queue.append((new_tuple, path + [move_name]))

    return None  # No solution found within depth
```

### 3. Iterative Deepening A* (IDA*)

```python
def ida_star_solve(puzzle, heuristic_fn):
    """IDA* search with heuristic"""

    def search(state, g, bound, path):
        f = g + heuristic_fn(state, puzzle.goal)
        if f > bound:
            return f, None

        if puzzle.is_solved(state):
            return -1, path

        min_bound = float('inf')

        for move_name in puzzle.moves:
            # Prune inverse of last move
            if path and puzzle.inverse_moves.get(path[-1]) == move_name:
                continue

            new_state = puzzle.apply_move(state, move_name)
            new_path = path + [move_name]

            result, solution = search(new_state, g + 1, bound, new_path)

            if result == -1:
                return -1, solution
            if result < min_bound:
                min_bound = result

        return min_bound, None

    bound = heuristic_fn(puzzle.initial, puzzle.goal)
    state = puzzle.initial.copy()

    while True:
        result, solution = search(state, 0, bound, [])
        if result == -1:
            return solution
        if result == float('inf'):
            return None
        bound = result
```

### 4. Kociemba Algorithm for Cube Puzzles

```python
import kociemba

def solve_cube_puzzle(puzzle):
    """Use Kociemba's algorithm for cube puzzles"""

    # Convert to standard cube notation
    cube_string = convert_to_kociemba_format(puzzle.initial)

    # Solve using two-phase algorithm
    solution = kociemba.solve(cube_string)

    # Convert back to competition format
    moves = convert_moves_to_competition_format(solution)

    return moves

# For larger cube variants, may need custom implementation
def two_phase_solve(puzzle):
    """Two-phase algorithm:
    Phase 1: Reduce to a smaller subgroup
    Phase 2: Solve within subgroup
    """
    # Phase 1: Orient edges and corners, reduce to <U, D, L2, R2, F2, B2>
    phase1_solution = solve_phase1(puzzle)

    # Apply phase 1
    intermediate = puzzle.apply_sequence(puzzle.initial, phase1_solution)

    # Phase 2: Solve remaining permutation
    phase2_solution = solve_phase2(intermediate, puzzle.goal)

    return phase1_solution + phase2_solution
```

### 5. Beam Search for Large Puzzles

```python
import heapq

def beam_search(puzzle, beam_width=1000, max_depth=100):
    """Beam search with limited frontier"""

    # Priority queue: (negative_progress, depth, state, path)
    initial_score = -compute_progress(puzzle.initial, puzzle.goal)
    beam = [(initial_score, 0, tuple(puzzle.initial), [])]

    best_solution = None
    best_length = float('inf')

    for depth in range(max_depth):
        next_beam = []

        for _, d, state_tuple, path in beam:
            state = np.array(state_tuple)

            for move_name in puzzle.moves:
                new_state = puzzle.apply_move(state, move_name)
                new_path = path + [move_name]

                if puzzle.is_solved(new_state):
                    if len(new_path) < best_length:
                        best_solution = new_path
                        best_length = len(new_path)
                    continue

                score = -compute_progress(new_state, puzzle.goal)
                next_beam.append((score, d + 1, tuple(new_state), new_path))

        # Keep top beam_width candidates
        beam = heapq.nsmallest(beam_width, next_beam)

        if not beam:
            break

    return best_solution

def compute_progress(state, goal):
    """Heuristic: count matching positions"""
    return np.sum(state == goal)
```

### 6. Local Search Optimization

```python
def optimize_solution(puzzle, solution):
    """Optimize an existing solution by removing redundant moves"""
    optimized = solution.copy()
    improved = True

    while improved:
        improved = False

        # Try removing pairs of inverse moves
        i = 0
        while i < len(optimized) - 1:
            if puzzle.inverse_moves.get(optimized[i]) == optimized[i + 1]:
                optimized = optimized[:i] + optimized[i+2:]
                improved = True
            else:
                i += 1

        # Try replacing subsequences with shorter equivalents
        for length in range(2, min(6, len(optimized))):
            for start in range(len(optimized) - length + 1):
                subsequence = optimized[start:start + length]
                replacement = find_shorter_equivalent(puzzle, subsequence)
                if replacement and len(replacement) < length:
                    optimized = optimized[:start] + replacement + optimized[start + length:]
                    improved = True
                    break

    return optimized
```

---

## Technical Insights

### Puzzle Complexity

| Puzzle Type | State Space | Typical Solution Length |
|-------------|-------------|------------------------|
| Cube 2/2/2 | 3.7 million | ~11 moves |
| Cube 3/3/3 | 4.3 × 10^19 | ~20 moves (optimal) |
| Globe | Variable | 10-50 moves |
| Wreath | Variable | 5-30 moves |

### Algorithm Selection

| Algorithm | Best For | Complexity |
|-----------|----------|------------|
| BFS | Small puzzles | O(b^d) |
| IDA* | Medium puzzles with good heuristic | O(b^d) |
| Kociemba | Standard cubes | Near-optimal |
| Beam search | Large puzzles | Approximate |

### Heuristics

| Heuristic | Description | Admissible? |
|-----------|-------------|-------------|
| Manhattan distance | Sum of position distances | Yes |
| Pattern databases | Precomputed subproblem solutions | Yes |
| Matching count | Simple but weak | No |

---

## Solution Links

| Place | Solution |
|-------|----------|
| 1st | [Discussion](https://www.kaggle.com/competitions/santa-2023/discussion/470517) |
| 2nd | [Discussion](https://www.kaggle.com/competitions/santa-2023/discussion/470553) |
| 3rd | [Discussion](https://www.kaggle.com/competitions/santa-2023/discussion/470483) |
| 4th | [Discussion](https://www.kaggle.com/competitions/santa-2023/discussion/470427) |
| 5th | [Discussion](https://www.kaggle.com/competitions/santa-2023/discussion/470531) |

---

## Citation

```bibtex
@misc{santa-2023,
    author = {Kaggle},
    title = {Santa 2023 - The Polytope Permutation Puzzle},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/santa-2023}},
    note = {Kaggle}
}
```
