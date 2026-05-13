"""
Data models for the scheduling optimization API.
Using Pydantic v2 for validation & automatic OpenAPI schema generation.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Core problem definition
# ---------------------------------------------------------------------------


class Resource(BaseModel):
    """A renewable resource (machine, worker pool, etc.)."""

    id: str = Field(..., description="Unique resource identifier.")
    name: str = Field(default="", description="Human-readable name.")
    capacity: int = Field(
        ..., ge=1, description="Maximum simultaneous usage at any time unit."
    )
    cost_per_unit_time: float = Field(
        default=0.0, ge=0.0, description="Cost charged per time unit of usage."
    )


class Task(BaseModel):
    """A schedulable task / activity."""

    id: str = Field(..., description="Unique task identifier.")
    name: str = Field(default="", description="Human-readable name.")
    duration: int = Field(..., ge=1, description="Processing time in time units.")
    resource_requirements: dict[str, int] = Field(
        default_factory=dict,
        description="Resource id -> units needed concurrently for the full duration.",
    )
    predecessors: list[str] = Field(
        default_factory=list,
        description="Task ids that must complete before this task can start.",
    )
    release_time: int = Field(
        default=0, ge=0, description="Earliest possible start time."
    )
    deadline: int | None = Field(
        default=None,
        description="Latest allowed completion time (hard constraint if set).",
    )
    due_date: int | None = Field(
        default=None,
        description="Target completion time (soft - contributes to tardiness).",
    )
    weight: float = Field(
        default=1.0, ge=0.0, description="Priority weight for objective terms."
    )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class ConstraintKind(str, Enum):
    PRECEDENCE = "precedence"                # A before B
    NO_OVERLAP = "no_overlap"                # A and B cannot run concurrently
    SAME_RESOURCE = "same_resource"          # A and B must use same resource
    MUTEX_RESOURCE = "mutex_resource"        # A and B must NOT share a resource
    TIME_WINDOW = "time_window"              # A must fall within [earliest, latest]
    MIN_GAP = "min_gap"                      # At least k time units between A end and B start
    MAX_GAP = "max_gap"                      # At most k time units between A end and B start
    RESOURCE_CAP = "resource_cap"            # Override capacity for a window
    CUSTOM = "custom"                        # Arbitrary user-defined (opaque to solver)


class Constraint(BaseModel):
    """A hard or soft constraint on the schedule."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: ConstraintKind
    hard: bool = Field(
        default=True,
        description="If True, solutions violating it are infeasible. "
        "If False, violations are allowed but penalised by `penalty_weight`.",
    )
    priority: int = Field(
        default=100,
        ge=0,
        description="Higher priority constraints win in conflict resolution.",
    )
    penalty_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost per unit violation for soft constraints.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific parameters (e.g. {'a': 'T1', 'b': 'T2', 'gap': 3}).",
    )
    description: str = Field(default="")


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class ObjectiveKind(str, Enum):
    MAKESPAN = "makespan"
    WEIGHTED_TARDINESS = "weighted_tardiness"
    TOTAL_COST = "total_cost"
    RESOURCE_LEVELING = "resource_leveling"
    MAX_UTILIZATION = "max_utilization"


class ObjectiveTerm(BaseModel):
    kind: ObjectiveKind
    weight: float = Field(default=1.0, ge=0.0)


# ---------------------------------------------------------------------------
# Problem & heuristic specification
# ---------------------------------------------------------------------------


class HeuristicKind(str, Enum):
    GREEDY_EST = "greedy_est"                # Earliest Start Time priority
    GREEDY_SPT = "greedy_spt"                # Shortest Processing Time
    GREEDY_MWKR = "greedy_mwkr"              # Most Work Remaining
    GREEDY_CRITICAL = "greedy_critical"      # Critical path based
    SIMULATED_ANNEALING = "simulated_annealing"
    TABU_SEARCH = "tabu_search"
    GENETIC = "genetic"
    CONSTRAINT_RELAXATION = "constraint_relaxation"


class HeuristicConfig(BaseModel):
    """Algorithm-specific knobs. Unknown keys are ignored by the engine."""

    # Shared
    random_seed: int | None = None
    time_limit_seconds: float = Field(default=10.0, ge=0.1)

    # SA
    sa_initial_temperature: float = 100.0
    sa_cooling_rate: float = 0.995
    sa_min_temperature: float = 0.01
    sa_iterations_per_temp: int = 50

    # Tabu
    tabu_tenure: int = 10
    tabu_max_iterations: int = 500
    tabu_neighborhood_size: int = 30

    # GA
    ga_population_size: int = 50
    ga_generations: int = 100
    ga_crossover_rate: float = 0.8
    ga_mutation_rate: float = 0.1
    ga_elitism: int = 2

    # Relaxation
    relax_soft_fraction: float = 0.3


class SchedulingProblem(BaseModel):
    """A complete problem instance submitted by the client."""

    id: str = Field(default_factory=lambda: f"prob_{uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    horizon: int = Field(
        default=0,
        ge=0,
        description="Planning horizon in time units. 0 means auto-compute from tasks.",
    )
    resources: list[Resource]
    tasks: list[Task]
    constraints: list[Constraint] = Field(default_factory=list)
    objectives: list[ObjectiveTerm] = Field(
        default_factory=lambda: [ObjectiveTerm(kind=ObjectiveKind.MAKESPAN)]
    )

    @field_validator("tasks")
    @classmethod
    def _unique_task_ids(cls, v: list[Task]) -> list[Task]:
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Task ids must be unique.")
        return v

    @field_validator("resources")
    @classmethod
    def _unique_resource_ids(cls, v: list[Resource]) -> list[Resource]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Resource ids must be unique.")
        return v


# ---------------------------------------------------------------------------
# Solve requests / responses
# ---------------------------------------------------------------------------


class SolveRequest(BaseModel):
    problem_id: str
    heuristic: HeuristicKind
    config: HeuristicConfig = Field(default_factory=HeuristicConfig)


class TaskAssignment(BaseModel):
    task_id: str
    start: int
    end: int
    resources_used: dict[str, int] = Field(default_factory=dict)


class SolutionQuality(BaseModel):
    feasible: bool
    makespan: int
    weighted_tardiness: float
    total_cost: float
    hard_violations: int
    soft_violation_penalty: float
    resource_utilization: dict[str, float]
    objective_value: float  # The value of the configured objective


class Solution(BaseModel):
    id: str = Field(default_factory=lambda: f"sol_{uuid4().hex[:8]}")
    problem_id: str
    heuristic: HeuristicKind
    assignments: list[TaskAssignment]
    quality: SolutionQuality
    wall_time_seconds: float
    iterations: int = 0
    log: list[str] = Field(default_factory=list)


class RankedSolutions(BaseModel):
    problem_id: str
    solutions: list[Solution]          # Sorted best-first by objective_value
    best_objective: float | None
    total_wall_time_seconds: float


# ---------------------------------------------------------------------------
# Simulation / scenario analysis
# ---------------------------------------------------------------------------


class ScenarioOverride(BaseModel):
    """Mutations applied on top of the base problem to create a scenario."""

    name: str
    resource_capacity_changes: dict[str, int] = Field(default_factory=dict)
    task_duration_multipliers: dict[str, float] = Field(default_factory=dict)
    add_tasks: list[Task] = Field(default_factory=list)
    remove_task_ids: list[str] = Field(default_factory=list)


class SimulationRequest(BaseModel):
    problem_id: str
    heuristics: list[HeuristicKind] = Field(default_factory=list)
    scenarios: list[ScenarioOverride] = Field(default_factory=list)
    config: HeuristicConfig = Field(default_factory=HeuristicConfig)


class ScenarioResult(BaseModel):
    scenario_name: str
    solutions: list[Solution]
    best_solution_id: str | None


class SimulationResult(BaseModel):
    problem_id: str
    scenarios: list[ScenarioResult]
    cross_scenario_summary: dict[str, Any]
    total_wall_time_seconds: float


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


class RecommendationRequest(BaseModel):
    problem_id: str
    time_budget_seconds: float = Field(default=30.0, ge=0.1)
    primary_objective: ObjectiveKind = ObjectiveKind.MAKESPAN
    prefer: Literal["speed", "quality", "balanced"] = "balanced"


class Recommendation(BaseModel):
    heuristic: HeuristicKind
    confidence: float                          # 0..1
    rationale: list[str]
    suggested_config: HeuristicConfig


class RecommendationResponse(BaseModel):
    problem_id: str
    problem_features: dict[str, float]
    ranked_recommendations: list[Recommendation]
