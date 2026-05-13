# Scheduling Optimization API

A production-ready FastAPI service for resource-constrained project scheduling with pluggable heuristics, constraint configuration, multi-scenario simulation, and a model-recommendation engine.

This is the companion software layer for the MILP / CP-SAT formulation developed in the production-scheduling analysis: it exposes the same problem structure (tasks, resources, precedence, changeovers, deadlines, soft/hard constraints) as a network service, and lets you experiment with heuristic solvers when exact solvers are too slow.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the API
uvicorn scheduling_api.main:app --reload --port 8000

# 3. Open interactive docs
#    Swagger UI : http://localhost:8000/docs
#    ReDoc      : http://localhost:8000/redoc

# 4. In another terminal, run the end-to-end demo
python -m scheduling_api.test_client
```

The test client creates a 12-task bakery production problem, adds a soft min-gap constraint, asks the recommender which heuristic to use, solves it, then runs a 4-scenario x 5-heuristic simulation and prints a ranked comparison.

## What's in the box

| File | Purpose |
|------|---------|
| `models.py` | Pydantic schemas for problems, constraints, heuristics, solutions |
| `constraints.py` | Pluggable constraint evaluation engine (hard/soft, 9 kinds) |
| `heuristics.py` | Greedy dispatch rules + SA + Tabu + GA + relaxation, all on SGS |
| `recommender.py` | Feature-based scoring of heuristics given problem + budget |
| `main.py` | FastAPI app wiring endpoints together |
| `test_client.py` | End-to-end demo (health → problem → solve → simulate) |
| `API_SPEC.md` | Full technical specification |

## Endpoints at a glance

```
POST   /api/v1/problems                   # Create
GET    /api/v1/problems/{pid}             # Retrieve
PUT    /api/v1/problems/{pid}             # Update
DELETE /api/v1/problems/{pid}             # Delete

POST   /api/v1/problems/{pid}/constraints # Attach constraint
DELETE /api/v1/problems/{pid}/constraints/{cid}

POST   /api/v1/solve                      # Run one heuristic
POST   /api/v1/simulate                   # Multi-heuristic, multi-scenario
POST   /api/v1/recommend                  # Rank heuristics by problem features

GET    /api/v1/solutions/{sid}            # One solution
GET    /api/v1/problems/{pid}/solutions   # Ranked solutions

GET    /api/v1/heuristics                 # Catalogue
GET    /api/v1/health
```

## Heuristics catalogue

| Heuristic | Class | Best for |
|-----------|-------|----------|
| `greedy_est` | Constructive | Release-time-driven problems, huge instances |
| `greedy_spt` | Constructive | Flow time / throughput |
| `greedy_mwkr` | Constructive | Makespan on job-shop-like problems |
| `greedy_critical` | Constructive | Honouring the longest precedence chain |
| `simulated_annealing` | Local search | Medium (30-500 tasks), balanced quality/speed |
| `tabu_search` | Local search | Structured problems with many constraints |
| `genetic` | Population | Multi-objective, rugged landscapes |
| `constraint_relaxation` | Constructive | Possibly-infeasible, tight problems |

## Design notes

- **Serial SGS backbone** — every heuristic reduces to exploring the space of activity priority lists, which the Serial Schedule Generation Scheme turns into feasible resource-respecting schedules. This keeps neighborhoods simple and comparable.
- **Infeasibility handling** — hard violations are not masked; they add a fixed penalty (`10,000` each) to the objective so infeasible solutions still compare sensibly but always lose to feasible ones.
- **Storage** — in-memory for demo simplicity. Replace `_problems` / `_solutions` dicts with your database layer.
- **Concurrency** — simulate calls are sequential per request; parallelise with a worker pool (Celery/RQ) when you have long-running heuristics.
- **No MIP dependency** — the code runs with pure Python + FastAPI. Hook in OR-Tools or Gurobi by adding a new `HeuristicKind.EXACT_*` and dispatching in `run_heuristic`.

See `API_SPEC.md` for the full technical specification.
