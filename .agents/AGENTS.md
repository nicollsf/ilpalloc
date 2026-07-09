# AI Primer: ilpalloc

## System Overview
`ilpalloc` is a Python utility for allocating students to academic projects. It translates a multi-variable assignment problem into an Integer Linear Program (ILP) and solves it using the `pulp` library (defaulting to the CBC solver).

## Core Files
- `ilpalloc.py`: The single executable script containing all parsing, optimization, and output logic.

## Data Structures & I/O
- **Input**: An Excel file loaded via `pandas`. Requires three sheets: `choices`, `limits`, and `scores`.
- **Output**: An Excel file (`*_results.xlsx`) containing `allocations` and `unallocated` sheets.
- **Internal Matrices**: Relies heavily on `numpy` arrays for vectorized indexing.
  - `tm`: Cost matrix `(m x n)` where `m` is students, `n` is unique projects.
  - `psupi`: Project-to-Supervisor mapping array.
  - `slimsn`: Max limits array mapped to supervisor indices.
  - `spsi`: Boolean array flagging self-proposed projects.

## Optimization Model (ILP Formulation)
- **Decision Variables**: $x_{i,j} \in \{0, 1\}$ mapped via `LpVariable.dicts` over students $i \in [0, m-1]$ and projects $j \in [0, n-1]$.
- **Objective Function**: Minimize the negative preference scores. Higher scores indicate a better preference match. A slight Gaussian noise ($\sim \mathcal{N}(0, 0.001)$) is added to the flattened cost matrix to break ties deterministically (seed fixed to `12345`).
- **Constraints**:
  1. **Student Limit**: $\sum_j x_{i,j} \le 1 \quad \forall i$ (max 1 project per student).
  2. **Project Limit**: $\sum_i x_{i,j} \le 1 \quad \forall j$ (max 1 student per project).
  3. **Supervisor Limit**: $\sum_{i} \sum_{j \in S_k} x_{i,j} \le L_k \quad \forall k$ (supervisor load limits, where $S_k$ is the set of projects belonging to supervisor $k$, and $L_k$ is their load limit).
  4. **Invalid Assignments**: $x_{i,j} = 0$ for any $i,j$ combination where the preference score is $\le 0$ (meaning the student did not select the project).

## Business Logic / Quirks to Note
1. **Supervisor Deduction**: The supervisor ID is extracted using Regex `^[A-Z]+` from the project code.
2. **Self-Proposed Projects**: Projects containing lowercase letters (Regex `[a-z]`) are considered self-proposed. The script specifically tracks metrics on these.
3. **Missing Scores**: If a student makes `N` choices but the `scores` sheet only provides weights for $< N$ choices, the script assumes a score of `0` for unmapped choices, which inherently prevents assignment to those choices due to the "Invalid Assignments" constraint.

## Dependencies
- `pandas`, `numpy`, `pulp`, `openpyxl`

## Potential Extension Points
- Integrating alternative solvers (e.g., Gurobi, CPLEX) if data scales significantly.
- Soft limits for supervisors (penalizing overloading rather than strictly forbidding it).
- Adding minimum project enrollment constraints (though currently moot since max students per project is 1).
