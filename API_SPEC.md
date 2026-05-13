# Scheduling Optimization API — Technical Specification v1.0

## 1. Overview

This service solves **Resource-Constrained Project Scheduling Problems (RCPSP)** and closely related variants (job-shop, flow-shop, batch processing with changeovers). It exposes a REST API that lets clients:

1. Define a scheduling problem (tasks, resources, constraints, objectives).
2. Ask a **recommender** which heuristic is most suitable given the problem features and time budget.
3. Solve with a chosen heuristic, or run a **simulation** across multiple heuristics and scenarios.
4. Retrieve **ranked solutions** with a standard quality record (makespan, tardiness, cost, utilisation, violations, objective value).

The API is heuristic-first but structurally compatible with the MILP / CP-SAT formulations produced by exact solvers — the same `Task`, `Resource`, and `Constraint` objects can be passed through to Gurobi / OR-Tools by adding a new heuristic kind.

## 2. Architecture

```
┌────────────────────────┐
│     FastAPI layer      │  Input validation (Pydantic), routing, persistence
│      (main.py)         │
└──────────┬─────────────┘
           │
 ┌─────────┴──────────┬──────────────┬──────────────┐
 │                    │              │              │
 ▼                    ▼              ▼              ▼
Constraints       Heuristics     Recommender    Storage
(constraints.py)  (heuristics.py)(recommender.py)(in-memory dict)
```

Every heuristic funnels through the same **Serial Schedule Generation Scheme (SGS)** which converts a priority list of tasks into a feasible schedule (respecting precedence + resource capacity). This means:

- Heuristics differ only in how they build / mutate the priority list.
- Adding a new local-search operator is a single function.
- Solution quality metrics are computed uniformly.

## 3. Data model

### 3.1 Resource

```json
{
  "id": "oven",
  "name": "Convection oven",
  "capacity": 2,
  "cost_per_unit_time": 15.0
}
```

Renewable resources only; capacity is respected at every time unit of the horizon. For non-renewable resources, encode as a cumulative inventory constraint using a `custom` constraint with pre-computed violation magnitude.

### 3.2 Task

```json
{
  "id": "R0_bake",
  "name": "Bake recipe 0",
  "duration": 5,
  "resource_requirements": {"oven": 1},
  "predecessors": ["R0_mix"],
  "release_time": 0,
  "deadline": 20,
  "due_date": 15,
  "weight": 2.0
}
```

- `deadline` is a **hard** constraint (end > deadline → infeasible).
- `due_date` is a **soft** constraint contributing to weighted tardiness.
- `weight` multiplies into tardiness terms.

### 3.3 Constraint kinds

Nine built-in kinds, configurable as hard or soft:

| Kind | Params | Meaning |
|------|--------|---------|
| `precedence` | `a`, `b` | a finishes before b starts |
| `no_overlap` | `a`, `b` | a and b never run concurrently |
| `same_resource` | `a`, `b` | must share ≥1 resource |
| `mutex_resource` | `a`, `b` | must NOT share resources |
| `time_window` | `a`, `earliest`, `latest` | a must sit inside window |
| `min_gap` | `a`, `b`, `gap` | ≥ gap units between a.end and b.start |
| `max_gap` | `a`, `b`, `gap` | ≤ gap units |
| `resource_cap` | `resource_id`, `start`, `end`, `capacity` | Override capacity in window |
| `custom` | `violation` | Caller-computed violation magnitude |

Every constraint accepts:
- `hard: bool` — True = infeasibility; False = `penalty_weight × magnitude` added to objective.
- `priority: int` — used by conflict resolution (higher wins; client-side).
- `penalty_weight: float` — cost per unit violation.

### 3.4 Objective

Multiple weighted terms; the solver minimises their weighted sum plus `10,000 × hard_violations + soft_penalty`.

Available kinds: `makespan`, `weighted_tardiness`, `total_cost`, `resource_leveling`, `max_utilization` (negated so higher is better).

## 4. REST endpoints

All endpoints are prefixed with `/api/v1`. Responses use standard HTTP codes (200/201/404/422).

### 4.1 Problem CRUD

```
POST   /problems                 → SchedulingProblem
GET    /problems                 → SchedulingProblem[]
GET    /problems/{pid}           → SchedulingProblem
PUT    /problems/{pid}           → SchedulingProblem
DELETE /problems/{pid}           → {status, id}
```

### 4.2 Constraints

```
POST   /problems/{pid}/constraints         body: Constraint
DELETE /problems/{pid}/constraints/{cid}
```

### 4.3 Solve

```http
POST /api/v1/solve
Content-Type: application/json

{
  "problem_id": "prob_abc12345",
  "heuristic": "simulated_annealing",
  "config": {
    "time_limit_seconds": 10,
    "random_seed": 42,
    "sa_initial_temperature": 100,
    "sa_cooling_rate": 0.995
  }
}
```

Response (`Solution`):

```json
{
  "id": "sol_7d2e...",
  "problem_id": "prob_abc12345",
  "heuristic": "simulated_annealing",
  "assignments": [
    {"task_id": "T1", "start": 0, "end": 3, "resources_used": {"M1": 1}}
  ],
  "quality": {
    "feasible": true,
    "makespan": 19,
    "weighted_tardiness": 0.0,
    "total_cost": 312.0,
    "hard_violations": 0,
    "soft_violation_penalty": 0.0,
    "resource_utilization": {"M1": 0.71, "M2": 0.52},
    "objective_value": 33.0
  },
  "wall_time_seconds": 2.01,
  "iterations": 26800,
  "log": ["SA start T=100.0 initial obj=48.0", "  it=42 T=..."]
}
```

### 4.4 Simulate

```http
POST /api/v1/simulate
{
  "problem_id": "prob_abc12345",
  "heuristics": ["greedy_critical", "simulated_annealing", "genetic"],
  "scenarios": [
    {"name": "baseline"},
    {"name": "broken_oven", "resource_capacity_changes": {"oven": 1}},
    {"name": "rush_job", "task_duration_multipliers": {"R0_bake": 0.6}}
  ],
  "config": {"time_limit_seconds": 2.0}
}
```

Response contains per-scenario ranked solutions and a cross-scenario summary naming the heuristic with the best average objective.

### 4.5 Recommend

```http
POST /api/v1/recommend
{
  "problem_id": "prob_abc12345",
  "time_budget_seconds": 5.0,
  "primary_objective": "makespan",
  "prefer": "balanced"
}
```

Returns the extracted problem features and a ranked list of heuristics with per-heuristic confidence, rationale, and a **pre-tuned configuration** tailored to the problem size and budget.

### 4.6 Solutions

```
GET /solutions/{sid}              → Solution
GET /problems/{pid}/solutions     → RankedSolutions (sorted by objective)
```

## 5. Heuristics specification

### 5.1 Greedy priority-dispatching (constructive)

Four variants differing only in how they order the activity list before passing it to SGS:

| Variant | Priority rule | Strength | Weakness |
|---------|---------------|----------|----------|
| `greedy_est` | `release_time` ascending | Very fast, good for release-time heavy problems | Ignores downstream work |
| `greedy_spt` | `duration` ascending | Minimises mean flow time | Poor for makespan |
| `greedy_mwkr` | "Most work remaining" (longest tail) descending | Good makespan on job-shop | O(n²) tail computation |
| `greedy_critical` | Longest path depth descending | Tracks critical chain | Blind to resource conflict |

**Complexity:** O(n log n) sort + O(n·H) SGS where H is the horizon in discrete units. Deterministic and reproducible.

**Use when:** you need a solution in milliseconds, or as a seed for local search.

### 5.2 Simulated Annealing (local search)

- **Initial solution:** `greedy_critical`.
- **Neighborhood:** adjacent swap in the priority list (rejected if it violates precedence).
- **Acceptance:** better moves always; worse with probability `exp(-Δ/T)`.
- **Cooling:** geometric, `T ← T·α` (α default 0.995) until `T_min`.

**Parameters:**
```json
{
  "sa_initial_temperature": 100.0,
  "sa_cooling_rate": 0.995,
  "sa_min_temperature": 0.01,
  "sa_iterations_per_temp": 50
}
```

**Use when:** 30–500 tasks, moderate time budget (5–60s), mixed hard/soft constraints, single objective.

**Tradeoffs:** simple to tune, but may require multiple restarts for rugged landscapes.

### 5.3 Tabu Search (local search)

- **Initial solution:** `greedy_mwkr`.
- **Neighborhood:** sampled adjacent swaps (`tabu_neighborhood_size` candidates).
- **Memory:** tabu list of recent `(a,b)` swap moves, fixed length (`tabu_tenure`).
- **Aspiration:** tabu moves are allowed if they improve the best-so-far.

**Parameters:**
```json
{
  "tabu_tenure": 10,
  "tabu_max_iterations": 500,
  "tabu_neighborhood_size": 30
}
```

**Use when:** you have many structured constraints; the tabu list avoids revisiting recent configurations. Often outperforms SA on job-shop-like problems with 50–800 tasks.

**Tradeoffs:** more parameters to tune; neighborhood cost dominates.

### 5.4 Genetic Algorithm (population)

- **Encoding:** permutation of task ids.
- **Seeding:** 4 greedy variants + random permutations (precedence-repaired).
- **Crossover:** Order Crossover (OX).
- **Mutation:** random swap with probability `ga_mutation_rate`.
- **Selection:** tournament (k=3), with elitism.

**Parameters:**
```json
{
  "ga_population_size": 50,
  "ga_generations": 100,
  "ga_crossover_rate": 0.8,
  "ga_mutation_rate": 0.1,
  "ga_elitism": 2
}
```

**Use when:** 100+ tasks, multi-objective, time budget ≥ 20s, or landscape is rugged.

**Tradeoffs:** slow convergence; needs enough evaluations to pay off.

### 5.5 Constraint Relaxation (constructive)

Tries all four greedy variants in sequence, returning the best. Acts as a safety net when one rule may be misled by a particular feature of the instance (e.g., heavy soft constraints). Extendable to a full **fix-and-relax** matheuristic if an exact solver is plugged in.

## 6. Model recommendation engine

### 6.1 Extracted features

| Feature | Formula |
|---------|---------|
| `n_tasks` | \|T\| |
| `n_resources` | \|R\| |
| `n_constraints` | \|C\| |
| `precedence_density` | total edges / n_tasks |
| `demand_ratio` | Σ demand / (Σ capacity × n_tasks) |
| `duration_cv` | σ(duration) / μ(duration) |
| `deadline_pressure` | n_tasks_with_deadline / n_tasks |
| `soft_fraction` | soft constraints / total constraints |

### 6.2 Scoring rules (abridged)

For each heuristic a baseline score is adjusted by the features and the user's `prefer` (`speed` / `quality` / `balanced`) + `time_budget_seconds`. Key rules:

- **Greedy** boosted when n > 500 or budget < 2s, with per-rule objective bias.
- **Simulated Annealing** boosted for 30 ≤ n ≤ 500 and balanced preference.
- **Tabu Search** boosted for constraint-heavy instances (`n_constraints > 10`).
- **Genetic** boosted for n ≥ 100 and budget ≥ 20s; penalised for very tight budgets.
- **Constraint Relaxation** boosted for `soft_fraction > 0.3` or `demand_ratio > 0.9`.

Scores are normalised to confidence ∈ [0, 1] per response. Suggested configurations are also tuned per heuristic and instance size (e.g., `tabu_tenure = √n`, `ga_population = min(120, n)`).

### 6.3 Extending

The rule set lives in `recommender._score_heuristics`. Replace it with a learned model (gradient-boosted tree, small neural net) by training on a corpus of solved problems and keeping the same `(features, request) → [(heuristic, score, rationale)]` interface.

## 7. Simulation & optimisation workflow

```
          ┌────────────┐
          │  Problem   │
          └─────┬──────┘
                │
    ┌───────────▼────────────┐
    │ Apply scenario deltas  │  (per scenario)
    └───────────┬────────────┘
                │
    ┌───────────▼────────────┐
    │ Run each heuristic     │  (per heuristic)
    │ against modified prob  │
    └───────────┬────────────┘
                │
    ┌───────────▼────────────┐
    │ Compute quality metrics│
    └───────────┬────────────┘
                │
    ┌───────────▼────────────┐
    │ Rank within scenario,  │
    │ aggregate across       │
    └───────────┬────────────┘
                │
                ▼
        SimulationResult
```

**Cross-scenario summary** reports per-scenario best heuristic plus the heuristic with lowest *average* objective across all scenarios — useful for "which algorithm should I use in production?" decisions when facing uncertain inputs.

## 8. Extension points

1. **Exact solvers** — Add `HeuristicKind.EXACT_MIP` and dispatch to Gurobi / CPLEX / HiGHS / OR-Tools CP-SAT inside `run_heuristic`. The same `SchedulingProblem` Pydantic model feeds the MILP.
2. **Persistence** — Replace the in-memory `_problems` / `_solutions` dicts with SQLAlchemy + Postgres or Redis. The storage layer is isolated to `main.py`.
3. **Async execution** — Long-running solves should be offloaded to a job queue (Celery / RQ / Dramatiq). Return `202 Accepted` + polling URL instead of blocking.
4. **Authentication** — Add JWT middleware; associate `SchedulingProblem.owner_id`.
5. **Rolling-horizon** — For multi-period plans, wrap `/solve` in an outer loop that advances time, fixes completed assignments, and resolves the remainder.
6. **Warm start** — The recommender returns a `suggested_config`; adding a `seed_solution_id` field to `SolveRequest` lets a heuristic start from an existing solution.
7. **Custom constraint kinds** — Add an entry to `ConstraintKind` and a branch in `constraints._evaluate_single`. Heuristics require no changes.

## 9. Example: end-to-end curl

```bash
# Create problem
PID=$(curl -s -X POST http://localhost:8000/api/v1/problems \
  -H "Content-Type: application/json" \
  -d '{
    "name": "toy",
    "resources": [{"id": "M", "capacity": 1}],
    "tasks": [
      {"id": "A", "duration": 3, "resource_requirements": {"M": 1}},
      {"id": "B", "duration": 2, "resource_requirements": {"M": 1}, "predecessors": ["A"]}
    ]
  }' | python -c "import sys, json; print(json.load(sys.stdin)['id'])")

# Recommend
curl -s -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d "{\"problem_id\": \"$PID\", \"time_budget_seconds\": 5.0, \"primary_objective\": \"makespan\"}" \
  | python -m json.tool

# Solve
curl -s -X POST http://localhost:8000/api/v1/solve \
  -H "Content-Type: application/json" \
  -d "{\"problem_id\": \"$PID\", \"heuristic\": \"greedy_critical\"}" \
  | python -m json.tool
```

## 10. Performance notes

The pure-Python SGS used by all heuristics is O(n × H) per schedule evaluation where H is the discrete horizon. Measured on a 2023-era laptop:

| n_tasks | Heuristic | Evaluations/sec |
|---------|-----------|-----------------|
| 12 | SA | ~13,000 |
| 50 | SA | ~2,500 |
| 200 | SA | ~400 |

For n > 500, the horizon grows and SGS dominates runtime. Accelerate by:

- Caching resource profiles across swap moves (incremental update instead of rebuild).
- Replacing dict-of-list profiles with NumPy arrays.
- Moving SGS to Cython / Rust (PyO3) — realistic 20–50× speedup.

Commercial solvers (Gurobi, CPLEX) will beat the heuristics on small instances (< 100 tasks) with loose constraints. Heuristics begin to win once the MILP's branch-and-bound tree explodes — typically n > 200 with tight resource contention.

## 11. Known limitations

- Preemption is not supported (tasks are non-interruptible).
- Only renewable resources; non-renewable require `custom` constraints.
- Changeover times between product families (as in the MILP analysis) are not directly modelled — they can be encoded as `min_gap` constraints between pairs, but a first-class `changeover_matrix` parameter is a planned extension.
- Single-machine storage: do not deploy as-is with >1 worker process without fronting `_problems` / `_solutions` with Redis.
