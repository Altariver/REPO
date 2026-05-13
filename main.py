"""
FastAPI application exposing the scheduling optimization service.

Endpoints (all prefixed with /api/v1):

  POST   /problems                    Create a scheduling problem
  GET    /problems                    List problems
  GET    /problems/{pid}              Retrieve a problem
  PUT    /problems/{pid}              Replace a problem
  DELETE /problems/{pid}              Delete

  POST   /problems/{pid}/constraints  Add a constraint
  DELETE /problems/{pid}/constraints/{cid}  Remove one

  POST   /solve                       Run a single heuristic
  POST   /simulate                    Multi-heuristic / multi-scenario

  POST   /recommend                   Rank heuristics for this problem

  GET    /solutions/{sid}             Fetch a solution
  GET    /problems/{pid}/solutions    List all solutions for a problem

  GET    /health                      Liveness probe
  GET    /heuristics                  List available heuristics with notes

Run:
    uvicorn scheduling_api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs         (Swagger UI)
    http://localhost:8000/redoc        (ReDoc)
"""
from __future__ import annotations

import copy
import time
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .heuristics import compute_quality, run_heuristic
from .models import (
    Constraint,
    HeuristicConfig,
    HeuristicKind,
    RankedSolutions,
    RecommendationRequest,
    RecommendationResponse,
    ScenarioOverride,
    ScenarioResult,
    SchedulingProblem,
    SimulationRequest,
    SimulationResult,
    SolveRequest,
    Solution,
)
from .recommender import recommend

# ---------------------------------------------------------------------------
# In-memory storage (swap for Postgres/Redis in production)
# ---------------------------------------------------------------------------

_problems: dict[str, SchedulingProblem] = {}
_solutions: dict[str, Solution] = {}
_lock = Lock()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Scheduling Optimization API",
    version="1.0.0",
    description=(
        "Production-grade scheduling API with pluggable heuristics (greedy "
        "dispatch rules, simulated annealing, tabu search, genetic algorithm, "
        "constraint relaxation), constraint configuration, multi-scenario "
        "simulation, and a model-recommendation engine."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Problem CRUD
# ---------------------------------------------------------------------------


@app.post("/api/v1/problems", response_model=SchedulingProblem, tags=["Problems"])
def create_problem(problem: SchedulingProblem) -> SchedulingProblem:
    with _lock:
        _problems[problem.id] = problem
    return problem


@app.get("/api/v1/problems", response_model=list[SchedulingProblem], tags=["Problems"])
def list_problems() -> list[SchedulingProblem]:
    return list(_problems.values())


@app.get("/api/v1/problems/{pid}", response_model=SchedulingProblem, tags=["Problems"])
def get_problem(pid: str) -> SchedulingProblem:
    if pid not in _problems:
        raise HTTPException(404, f"Problem {pid} not found")
    return _problems[pid]


@app.put("/api/v1/problems/{pid}", response_model=SchedulingProblem, tags=["Problems"])
def update_problem(pid: str, problem: SchedulingProblem) -> SchedulingProblem:
    if pid not in _problems:
        raise HTTPException(404, f"Problem {pid} not found")
    problem.id = pid
    with _lock:
        _problems[pid] = problem
    return problem


@app.delete("/api/v1/problems/{pid}", tags=["Problems"])
def delete_problem(pid: str) -> dict[str, str]:
    with _lock:
        if pid not in _problems:
            raise HTTPException(404, f"Problem {pid} not found")
        del _problems[pid]
    return {"status": "deleted", "id": pid}


# ---------------------------------------------------------------------------
# Constraint management
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/problems/{pid}/constraints",
    response_model=SchedulingProblem,
    tags=["Constraints"],
)
def add_constraint(pid: str, constraint: Constraint) -> SchedulingProblem:
    if pid not in _problems:
        raise HTTPException(404, f"Problem {pid} not found")
    with _lock:
        _problems[pid].constraints.append(constraint)
    return _problems[pid]


@app.delete(
    "/api/v1/problems/{pid}/constraints/{cid}",
    response_model=SchedulingProblem,
    tags=["Constraints"],
)
def remove_constraint(pid: str, cid: str) -> SchedulingProblem:
    if pid not in _problems:
        raise HTTPException(404, f"Problem {pid} not found")
    with _lock:
        before = len(_problems[pid].constraints)
        _problems[pid].constraints = [
            c for c in _problems[pid].constraints if c.id != cid
        ]
        if len(_problems[pid].constraints) == before:
            raise HTTPException(404, f"Constraint {cid} not found")
    return _problems[pid]


# ---------------------------------------------------------------------------
# Solve / simulate
# ---------------------------------------------------------------------------


def _persist_solution(sol: Solution) -> None:
    with _lock:
        _solutions[sol.id] = sol


def _solve_one(
    problem: SchedulingProblem,
    heuristic: HeuristicKind,
    config: HeuristicConfig,
) -> Solution:
    t0 = time.time()
    out = run_heuristic(problem, heuristic, config)
    elapsed = time.time() - t0
    quality = compute_quality(problem, out.assignments)
    sol = Solution(
        problem_id=problem.id,
        heuristic=heuristic,
        assignments=out.assignments,
        quality=quality,
        wall_time_seconds=round(elapsed, 4),
        iterations=out.iterations,
        log=out.log,
    )
    _persist_solution(sol)
    return sol


@app.post("/api/v1/solve", response_model=Solution, tags=["Solve"])
def solve(req: SolveRequest) -> Solution:
    if req.problem_id not in _problems:
        raise HTTPException(404, f"Problem {req.problem_id} not found")
    problem = _problems[req.problem_id]
    return _solve_one(problem, req.heuristic, req.config)


@app.post("/api/v1/simulate", response_model=SimulationResult, tags=["Solve"])
def simulate(req: SimulationRequest) -> SimulationResult:
    if req.problem_id not in _problems:
        raise HTTPException(404, f"Problem {req.problem_id} not found")
    base = _problems[req.problem_id]

    heuristics = req.heuristics or [
        HeuristicKind.GREEDY_CRITICAL,
        HeuristicKind.SIMULATED_ANNEALING,
        HeuristicKind.TABU_SEARCH,
    ]
    scenarios = req.scenarios or [ScenarioOverride(name="base")]

    t0 = time.time()
    scenario_results: list[ScenarioResult] = []

    for scenario in scenarios:
        modified = _apply_scenario(base, scenario)
        solutions: list[Solution] = []
        for h in heuristics:
            sol = _solve_one(modified, h, req.config)
            solutions.append(sol)
        solutions.sort(key=lambda s: s.quality.objective_value)
        scenario_results.append(
            ScenarioResult(
                scenario_name=scenario.name,
                solutions=solutions,
                best_solution_id=solutions[0].id if solutions else None,
            )
        )

    summary = _summarise_scenarios(scenario_results)
    return SimulationResult(
        problem_id=req.problem_id,
        scenarios=scenario_results,
        cross_scenario_summary=summary,
        total_wall_time_seconds=round(time.time() - t0, 4),
    )


def _apply_scenario(
    base: SchedulingProblem, scenario: ScenarioOverride
) -> SchedulingProblem:
    modified = copy.deepcopy(base)
    modified.id = f"{base.id}__{scenario.name}"
    # Resource capacity changes
    for r_id, new_cap in scenario.resource_capacity_changes.items():
        for r in modified.resources:
            if r.id == r_id:
                r.capacity = new_cap
    # Duration multipliers
    for t_id, mult in scenario.task_duration_multipliers.items():
        for t in modified.tasks:
            if t.id == t_id:
                t.duration = max(1, int(round(t.duration * mult)))
    # Add tasks
    modified.tasks.extend(scenario.add_tasks)
    # Remove tasks
    if scenario.remove_task_ids:
        to_remove = set(scenario.remove_task_ids)
        modified.tasks = [t for t in modified.tasks if t.id not in to_remove]
    return modified


def _summarise_scenarios(results: list[ScenarioResult]) -> dict[str, Any]:
    if not results:
        return {}
    summary: dict[str, Any] = {
        "by_scenario": {},
        "best_heuristic_overall": None,
    }
    heuristic_scores: dict[str, list[float]] = {}
    for sr in results:
        if not sr.solutions:
            continue
        best = sr.solutions[0]
        summary["by_scenario"][sr.scenario_name] = {
            "best_heuristic": best.heuristic.value,
            "best_objective": best.quality.objective_value,
            "best_makespan": best.quality.makespan,
            "feasible": best.quality.feasible,
        }
        for s in sr.solutions:
            heuristic_scores.setdefault(s.heuristic.value, []).append(
                s.quality.objective_value
            )
    # Heuristic with best average rank
    averages = {h: sum(v) / len(v) for h, v in heuristic_scores.items()}
    if averages:
        summary["best_heuristic_overall"] = min(averages, key=averages.get)
        summary["average_objective_by_heuristic"] = {
            h: round(v, 3) for h, v in averages.items()
        }
    return summary


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendation"],
)
def recommend_endpoint(req: RecommendationRequest) -> RecommendationResponse:
    if req.problem_id not in _problems:
        raise HTTPException(404, f"Problem {req.problem_id} not found")
    return recommend(_problems[req.problem_id], req)


# ---------------------------------------------------------------------------
# Solutions
# ---------------------------------------------------------------------------


@app.get("/api/v1/solutions/{sid}", response_model=Solution, tags=["Solutions"])
def get_solution(sid: str) -> Solution:
    if sid not in _solutions:
        raise HTTPException(404, f"Solution {sid} not found")
    return _solutions[sid]


@app.get(
    "/api/v1/problems/{pid}/solutions",
    response_model=RankedSolutions,
    tags=["Solutions"],
)
def list_solutions(pid: str) -> RankedSolutions:
    if pid not in _problems:
        raise HTTPException(404, f"Problem {pid} not found")
    sols = [s for s in _solutions.values() if s.problem_id == pid]
    sols.sort(key=lambda s: s.quality.objective_value)
    total_time = sum(s.wall_time_seconds for s in sols)
    return RankedSolutions(
        problem_id=pid,
        solutions=sols,
        best_objective=sols[0].quality.objective_value if sols else None,
        total_wall_time_seconds=round(total_time, 4),
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", tags=["Meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/heuristics", tags=["Meta"])
def list_heuristics() -> list[dict[str, str]]:
    return [
        {
            "kind": HeuristicKind.GREEDY_EST.value,
            "class": "constructive",
            "best_for": "huge problems, tight time budgets, release-time driven",
            "complexity": "O(n log n)",
        },
        {
            "kind": HeuristicKind.GREEDY_SPT.value,
            "class": "constructive",
            "best_for": "minimising average flow time & throughput",
            "complexity": "O(n log n)",
        },
        {
            "kind": HeuristicKind.GREEDY_MWKR.value,
            "class": "constructive",
            "best_for": "makespan minimisation (classic job-shop heuristic)",
            "complexity": "O(n^2)",
        },
        {
            "kind": HeuristicKind.GREEDY_CRITICAL.value,
            "class": "constructive",
            "best_for": "respecting the longest path / critical chain",
            "complexity": "O(n + e)",
        },
        {
            "kind": HeuristicKind.SIMULATED_ANNEALING.value,
            "class": "local_search",
            "best_for": "medium (30-500 tasks), balanced quality/speed",
            "complexity": "configurable (iterations x neighborhood)",
        },
        {
            "kind": HeuristicKind.TABU_SEARCH.value,
            "class": "local_search",
            "best_for": "structured problems with many constraints (50-800 tasks)",
            "complexity": "iterations x neighborhood size",
        },
        {
            "kind": HeuristicKind.GENETIC.value,
            "class": "population",
            "best_for": "multi-objective, large problems, rugged landscapes",
            "complexity": "generations x population size",
        },
        {
            "kind": HeuristicKind.CONSTRAINT_RELAXATION.value,
            "class": "constructive",
            "best_for": "possibly-infeasible or tightly-constrained problems",
            "complexity": "linear in alternatives tried",
        },
    ]
