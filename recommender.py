"""
Model recommendation engine.

Computes a feature vector from the problem instance and uses a rule-based
scoring model to rank heuristics by expected suitability. The rules are
tunable via the WEIGHTS table below.
"""
from __future__ import annotations

from statistics import mean, stdev

from .models import (
    HeuristicConfig,
    HeuristicKind,
    ObjectiveKind,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
    SchedulingProblem,
)


# ---------------------------------------------------------------------------
# Problem feature extraction
# ---------------------------------------------------------------------------


def extract_features(problem: SchedulingProblem) -> dict[str, float]:
    n_tasks = len(problem.tasks)
    n_resources = len(problem.resources)
    n_constraints = len(problem.constraints)

    # Precedence density
    total_edges = sum(len(t.predecessors) for t in problem.tasks)
    prec_density = total_edges / max(1, n_tasks)

    # Resource demand density
    total_req = sum(
        sum(t.resource_requirements.values()) for t in problem.tasks
    )
    total_cap = sum(r.capacity for r in problem.resources) * max(1, n_tasks)
    demand_ratio = total_req / max(1, total_cap)

    # Duration variance
    durations = [t.duration for t in problem.tasks]
    dur_mean = mean(durations) if durations else 0
    dur_cv = (stdev(durations) / dur_mean) if len(durations) > 1 and dur_mean else 0.0

    # Deadline pressure
    deadlines = [t.deadline for t in problem.tasks if t.deadline is not None]
    deadline_pressure = len(deadlines) / max(1, n_tasks)

    # Soft-constraint fraction
    soft = sum(1 for c in problem.constraints if not c.hard)
    soft_fraction = soft / max(1, n_constraints) if n_constraints else 0.0

    return {
        "n_tasks": float(n_tasks),
        "n_resources": float(n_resources),
        "n_constraints": float(n_constraints),
        "precedence_density": prec_density,
        "demand_ratio": demand_ratio,
        "duration_cv": dur_cv,
        "deadline_pressure": deadline_pressure,
        "soft_fraction": soft_fraction,
    }


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------


def _score_heuristics(
    features: dict[str, float],
    req: RecommendationRequest,
) -> list[tuple[HeuristicKind, float, list[str]]]:
    """
    Produce (heuristic, score, rationale-lines) tuples, higher score = better.
    Scores are relative; we normalise them in the response.
    """
    n = features["n_tasks"]
    budget = req.time_budget_seconds
    obj = req.primary_objective
    prefer = req.prefer

    results: list[tuple[HeuristicKind, float, list[str]]] = []

    # --- Greedy variants --------------------------------------------------
    greedy_base = 0.6
    if prefer == "speed":
        greedy_base += 0.3
    if n > 500:
        greedy_base += 0.3
    if budget < 2:
        greedy_base += 0.4

    rationale_common = [
        f"Problem has {int(n)} tasks, time budget is {budget:.1f}s.",
        "Greedy priority dispatching is O(n log n) and near-instant.",
    ]

    # Per-rule bias by objective
    if obj == ObjectiveKind.MAKESPAN:
        results.append((
            HeuristicKind.GREEDY_CRITICAL,
            greedy_base + 0.3,
            rationale_common + [
                "Critical-path rule tracks the longest chain, aligning with "
                "makespan minimisation."
            ],
        ))
        results.append((
            HeuristicKind.GREEDY_MWKR,
            greedy_base + 0.2,
            rationale_common + [
                "Most-Work-Remaining is a classic heuristic for makespan."
            ],
        ))
    elif obj == ObjectiveKind.WEIGHTED_TARDINESS:
        results.append((
            HeuristicKind.GREEDY_EST,
            greedy_base + 0.25,
            rationale_common + [
                "Earliest Start Time tends to reduce tardiness when release "
                "times and due dates matter."
            ],
        ))
    else:
        results.append((
            HeuristicKind.GREEDY_SPT,
            greedy_base + 0.1,
            rationale_common + [
                "Shortest Processing Time is a robust default when the "
                "objective is cost or utilisation driven."
            ],
        ))

    # --- Simulated Annealing ---------------------------------------------
    sa_score = 0.5
    if 30 <= n <= 500:
        sa_score += 0.3
    if prefer == "balanced":
        sa_score += 0.2
    if prefer == "quality":
        sa_score += 0.25
    if budget >= 5:
        sa_score += 0.2
    if features["precedence_density"] > 0.5:
        sa_score += 0.1  # SA copes well with heavy precedence via SGS
    results.append((
        HeuristicKind.SIMULATED_ANNEALING,
        sa_score,
        [
            "Simulated Annealing explores the neighbourhood via accepted "
            "worse moves, escaping local optima.",
            f"Recommended for medium problems (current n={int(n)}).",
            "Performs well with balanced time budgets (5-60s).",
        ],
    ))

    # --- Tabu Search ------------------------------------------------------
    tabu_score = 0.5
    if 50 <= n <= 800:
        tabu_score += 0.35
    if prefer == "quality":
        tabu_score += 0.25
    if features["n_constraints"] > 10:
        tabu_score += 0.15  # Tabu handles structured constraints well
    if budget >= 10:
        tabu_score += 0.15
    results.append((
        HeuristicKind.TABU_SEARCH,
        tabu_score,
        [
            "Tabu Search uses a short-term memory to avoid cycling; "
            "effective on structured scheduling problems.",
            f"Constraint count is {int(features['n_constraints'])} which "
            "benefits from memory-based search.",
        ],
    ))

    # --- Genetic Algorithm ------------------------------------------------
    ga_score = 0.4
    if n >= 100:
        ga_score += 0.25
    if len(_objectives_seen(req)) > 1:
        ga_score += 0.2  # multi-objective friendly
    if prefer == "quality":
        ga_score += 0.2
    if budget >= 20:
        ga_score += 0.2
    if budget < 3:
        ga_score -= 0.4  # GA needs many evaluations
    results.append((
        HeuristicKind.GENETIC,
        ga_score,
        [
            "Genetic Algorithm maintains a population with crossover & "
            "mutation - strong when multiple objectives or rugged "
            "landscapes are present.",
            "Requires many fitness evaluations; budget >=20s recommended.",
        ],
    ))

    # --- Constraint Relaxation -------------------------------------------
    cr_score = 0.35
    if features["soft_fraction"] > 0.3:
        cr_score += 0.3
    if features["demand_ratio"] > 0.9:
        cr_score += 0.25  # Tight capacity -> try relaxation
    if features["deadline_pressure"] > 0.5:
        cr_score += 0.15
    results.append((
        HeuristicKind.CONSTRAINT_RELAXATION,
        cr_score,
        [
            "Constraint relaxation constructs a feasible solution by "
            "relaxing soft constraints, then repairs.",
            f"Soft-constraint fraction {features['soft_fraction']:.2f} and "
            f"demand ratio {features['demand_ratio']:.2f}.",
        ],
    ))

    return results


def _objectives_seen(req: RecommendationRequest) -> set[str]:
    return {req.primary_objective.value}


# ---------------------------------------------------------------------------
# Config tuning per recommendation
# ---------------------------------------------------------------------------


def _suggested_config(
    heuristic: HeuristicKind,
    budget: float,
    features: dict[str, float],
) -> HeuristicConfig:
    cfg = HeuristicConfig(time_limit_seconds=max(0.5, budget))
    n = features["n_tasks"]

    if heuristic == HeuristicKind.SIMULATED_ANNEALING:
        cfg.sa_iterations_per_temp = max(20, int(min(200, n * 2)))
        cfg.sa_initial_temperature = max(10.0, n)
    elif heuristic == HeuristicKind.TABU_SEARCH:
        cfg.tabu_tenure = max(5, int(n ** 0.5))
        cfg.tabu_neighborhood_size = max(10, int(min(60, n)))
        cfg.tabu_max_iterations = max(100, int(budget * 50))
    elif heuristic == HeuristicKind.GENETIC:
        cfg.ga_population_size = max(20, int(min(120, n)))
        cfg.ga_generations = max(30, int(budget * 5))
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def recommend(
    problem: SchedulingProblem,
    req: RecommendationRequest,
) -> RecommendationResponse:
    features = extract_features(problem)
    scored = _score_heuristics(features, req)

    # Normalise scores to [0,1] for confidence
    max_s = max((s for _, s, _ in scored), default=1.0)
    min_s = min((s for _, s, _ in scored), default=0.0)
    spread = max_s - min_s or 1.0

    scored.sort(key=lambda x: -x[1])
    recs = [
        Recommendation(
            heuristic=h,
            confidence=round((s - min_s) / spread, 3),
            rationale=rat,
            suggested_config=_suggested_config(h, req.time_budget_seconds, features),
        )
        for h, s, rat in scored
    ]

    return RecommendationResponse(
        problem_id=problem.id,
        problem_features=features,
        ranked_recommendations=recs,
    )
