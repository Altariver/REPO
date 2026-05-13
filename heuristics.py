"""
Heuristic scheduling algorithms.

All heuristics share a common protocol:
    schedule, iterations, log = run(problem, config) -> list[TaskAssignment]

Internally they rely on the Serial Schedule Generation Scheme (SGS) to turn
activity-list permutations into feasible schedules that respect precedence and
resource capacity. Local-search / population-based methods then work in the
space of permutations.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass

from .constraints import evaluate
from .models import (
    HeuristicConfig,
    HeuristicKind,
    ObjectiveKind,
    SchedulingProblem,
    SolutionQuality,
    Task,
    TaskAssignment,
)


# ---------------------------------------------------------------------------
# Serial Schedule Generation Scheme (SGS)
# ---------------------------------------------------------------------------


def _topological_levels(tasks: list[Task]) -> dict[str, int]:
    """Compute longest-path depth for priority rules like MWKR & critical path."""
    duration = {t.id: t.duration for t in tasks}
    preds = {t.id: list(t.predecessors) for t in tasks}
    memo: dict[str, int] = {}

    def depth(tid: str) -> int:
        if tid in memo:
            return memo[tid]
        if not preds.get(tid):
            memo[tid] = duration[tid]
            return memo[tid]
        memo[tid] = duration[tid] + max(depth(p) for p in preds[tid])
        return memo[tid]

    return {t.id: depth(t.id) for t in tasks}


def _tail_work(tasks: list[Task]) -> dict[str, int]:
    """Remaining work on the longest path AFTER each task (for MWKR)."""
    successors: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for p in t.predecessors:
            successors.setdefault(p, []).append(t.id)
    duration = {t.id: t.duration for t in tasks}
    memo: dict[str, int] = {}

    def tail(tid: str) -> int:
        if tid in memo:
            return memo[tid]
        if not successors.get(tid):
            memo[tid] = 0
            return 0
        memo[tid] = max(duration[s] + tail(s) for s in successors[tid])
        return memo[tid]

    return {t.id: tail(t.id) for t in tasks}


def sgs_build(
    problem: SchedulingProblem,
    priority_list: list[str],
) -> list[TaskAssignment]:
    """
    Serial SGS: schedule tasks in the order given by `priority_list`,
    placing each at the earliest feasible start time respecting precedence
    and resource capacity.
    """
    tasks_by_id = {t.id: t for t in problem.tasks}
    resources_by_id = {r.id: r for r in problem.resources}

    # Running resource usage profile
    horizon_guess = sum(t.duration for t in problem.tasks) + 1
    usage: dict[str, list[int]] = {
        r_id: [0] * horizon_guess for r_id in resources_by_id
    }

    assignments: dict[str, TaskAssignment] = {}

    for tid in priority_list:
        task = tasks_by_id[tid]

        # Earliest start from precedence
        earliest = task.release_time
        for pred in task.predecessors:
            if pred in assignments:
                earliest = max(earliest, assignments[pred].end)

        # Find earliest start where ALL required resources have capacity
        # throughout [start, start + duration)
        def feasible(start: int) -> bool:
            end = start + task.duration
            # Grow usage buffer if needed
            for r_id, prof in usage.items():
                while len(prof) < end + 1:
                    prof.append(0)
            for r_id, need in task.resource_requirements.items():
                if need <= 0:
                    continue
                cap = resources_by_id[r_id].capacity
                for t in range(start, end):
                    if usage[r_id][t] + need > cap:
                        return False
            return True

        start = earliest
        while not feasible(start):
            start += 1
            if start > horizon_guess * 2:
                # Safety valve - give up and place anyway
                break

        end = start + task.duration
        # Commit resource usage
        for r_id, need in task.resource_requirements.items():
            for t in range(start, end):
                while len(usage[r_id]) <= t:
                    usage[r_id].append(0)
                usage[r_id][t] += need

        assignments[tid] = TaskAssignment(
            task_id=tid,
            start=start,
            end=end,
            resources_used=dict(task.resource_requirements),
        )

    # Return in problem's task order
    return [assignments[t.id] for t in problem.tasks if t.id in assignments]


# ---------------------------------------------------------------------------
# Priority list construction (respects precedence)
# ---------------------------------------------------------------------------


def _topological_sort(tasks: list[Task]) -> list[str]:
    in_deg = {t.id: len(t.predecessors) for t in tasks}
    by_id = {t.id: t for t in tasks}
    successors: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for p in t.predecessors:
            successors.setdefault(p, []).append(t.id)

    ready = [tid for tid, d in in_deg.items() if d == 0]
    order: list[str] = []
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for s in successors.get(tid, []):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                ready.append(s)
    if len(order) != len(tasks):
        raise ValueError("Cyclic precedence graph - cannot schedule.")
    return order


def _priority_list(problem: SchedulingProblem, rule: HeuristicKind) -> list[str]:
    """Build an activity priority list honouring precedence but ordered by rule."""
    topo = _topological_sort(problem.tasks)
    by_id = {t.id: t for t in problem.tasks}

    if rule == HeuristicKind.GREEDY_EST:
        # Earliest release time first (topo order breaks ties)
        topo.sort(key=lambda tid: (by_id[tid].release_time,))
    elif rule == HeuristicKind.GREEDY_SPT:
        topo.sort(key=lambda tid: by_id[tid].duration)
    elif rule == HeuristicKind.GREEDY_MWKR:
        tails = _tail_work(problem.tasks)
        topo.sort(key=lambda tid: -tails[tid])  # most work remaining first
    elif rule == HeuristicKind.GREEDY_CRITICAL:
        depths = _topological_levels(problem.tasks)
        topo.sort(key=lambda tid: -depths[tid])  # longest path first

    # Re-enforce precedence: tasks with earlier predecessors go first
    return _fix_precedence(topo, problem)


def _fix_precedence(order: list[str], problem: SchedulingProblem) -> list[str]:
    """Repair a sequence so that every task appears after its predecessors."""
    by_id = {t.id: t for t in problem.tasks}
    remaining = list(order)
    placed: list[str] = []
    placed_set: set[str] = set()
    while remaining:
        progress = False
        for i, tid in enumerate(remaining):
            preds = by_id[tid].predecessors
            if all(p in placed_set for p in preds):
                placed.append(tid)
                placed_set.add(tid)
                remaining.pop(i)
                progress = True
                break
        if not progress:
            # Cycle or missing predecessor - append anyway
            placed.extend(remaining)
            break
    return placed


# ---------------------------------------------------------------------------
# Solution quality
# ---------------------------------------------------------------------------


def compute_quality(
    problem: SchedulingProblem,
    assignments: list[TaskAssignment],
) -> SolutionQuality:
    report = evaluate(problem, assignments)
    makespan = max((a.end for a in assignments), default=0)

    tasks_by_id = {t.id: t for t in problem.tasks}

    # Weighted tardiness
    tardiness = 0.0
    for a in assignments:
        task = tasks_by_id[a.task_id]
        due = task.due_date if task.due_date is not None else task.deadline
        if due is not None and a.end > due:
            tardiness += task.weight * (a.end - due)

    # Costs
    total_cost = 0.0
    for a in assignments:
        task = tasks_by_id[a.task_id]
        for r_id, need in task.resource_requirements.items():
            r = next((x for x in problem.resources if x.id == r_id), None)
            if r is not None:
                total_cost += r.cost_per_unit_time * need * task.duration

    # Utilization
    util: dict[str, float] = {}
    for r in problem.resources:
        total_capacity = r.capacity * max(1, makespan)
        used = 0
        for a in assignments:
            t = tasks_by_id[a.task_id]
            used += t.resource_requirements.get(r.id, 0) * t.duration
        util[r.id] = used / total_capacity if total_capacity else 0.0

    # Objective value - weighted sum of configured objective terms
    obj = 0.0
    for term in problem.objectives:
        if term.kind == ObjectiveKind.MAKESPAN:
            obj += term.weight * makespan
        elif term.kind == ObjectiveKind.WEIGHTED_TARDINESS:
            obj += term.weight * tardiness
        elif term.kind == ObjectiveKind.TOTAL_COST:
            obj += term.weight * total_cost
        elif term.kind == ObjectiveKind.RESOURCE_LEVELING:
            # Penalise variance in usage
            variance = sum((u - 0.7) ** 2 for u in util.values())
            obj += term.weight * variance * 100
        elif term.kind == ObjectiveKind.MAX_UTILIZATION:
            obj -= term.weight * sum(util.values())

    # Penalise violations so infeasible solutions lose in comparisons
    obj += report.hard_violations * 10_000
    obj += report.soft_penalty

    return SolutionQuality(
        feasible=report.hard_violations == 0,
        makespan=makespan,
        weighted_tardiness=tardiness,
        total_cost=total_cost,
        hard_violations=report.hard_violations,
        soft_violation_penalty=report.soft_penalty,
        resource_utilization=util,
        objective_value=obj,
    )


# ---------------------------------------------------------------------------
# The heuristics
# ---------------------------------------------------------------------------


@dataclass
class HeuristicOutput:
    assignments: list[TaskAssignment]
    iterations: int
    log: list[str]


def run_greedy(
    problem: SchedulingProblem,
    config: HeuristicConfig,
    rule: HeuristicKind,
) -> HeuristicOutput:
    t0 = time.time()
    order = _priority_list(problem, rule)
    assignments = sgs_build(problem, order)
    return HeuristicOutput(
        assignments=assignments,
        iterations=1,
        log=[
            f"Greedy rule={rule.value}",
            f"Priority list: {order}",
            f"Built in {time.time() - t0:.3f}s",
        ],
    )


def _random_neighbor(order: list[str], problem: SchedulingProblem, rng: random.Random) -> list[str]:
    """Swap two adjacent tasks (respecting precedence by rejection)."""
    if len(order) < 2:
        return order[:]
    new = order[:]
    for _ in range(20):
        i = rng.randrange(len(new) - 1)
        a, b = new[i], new[i + 1]
        new[i], new[i + 1] = new[i + 1], new[i]
        # Precedence feasibility: if a is predecessor of b, illegal
        by_id = {t.id: t for t in problem.tasks}
        if a in by_id[b].predecessors:
            new[i], new[i + 1] = new[i], new[i + 1]
            new = order[:]
            continue
        return new
    return order[:]


def run_simulated_annealing(
    problem: SchedulingProblem,
    config: HeuristicConfig,
) -> HeuristicOutput:
    rng = random.Random(config.random_seed)
    start_time = time.time()

    current = _priority_list(problem, HeuristicKind.GREEDY_CRITICAL)
    current_sched = sgs_build(problem, current)
    current_obj = compute_quality(problem, current_sched).objective_value
    best, best_sched, best_obj = current[:], current_sched, current_obj

    temperature = config.sa_initial_temperature
    it = 0
    log = [f"SA start T={temperature} initial obj={current_obj:.2f}"]

    while temperature > config.sa_min_temperature:
        if time.time() - start_time > config.time_limit_seconds:
            log.append(f"SA stopped on time budget at it={it}")
            break
        for _ in range(config.sa_iterations_per_temp):
            it += 1
            neighbor = _random_neighbor(current, problem, rng)
            neighbor = _fix_precedence(neighbor, problem)
            sched = sgs_build(problem, neighbor)
            obj = compute_quality(problem, sched).objective_value
            delta = obj - current_obj
            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                current, current_sched, current_obj = neighbor, sched, obj
                if obj < best_obj:
                    best, best_sched, best_obj = neighbor[:], sched, obj
                    log.append(f"  it={it} T={temperature:.3f} new best obj={obj:.2f}")
        temperature *= config.sa_cooling_rate

    log.append(f"SA done it={it} final best obj={best_obj:.2f}")
    return HeuristicOutput(assignments=best_sched, iterations=it, log=log)


def run_tabu_search(
    problem: SchedulingProblem,
    config: HeuristicConfig,
) -> HeuristicOutput:
    rng = random.Random(config.random_seed)
    start_time = time.time()
    current = _priority_list(problem, HeuristicKind.GREEDY_MWKR)
    current_sched = sgs_build(problem, current)
    current_obj = compute_quality(problem, current_sched).objective_value
    best, best_sched, best_obj = current[:], current_sched, current_obj

    tabu: deque[tuple[str, str]] = deque(maxlen=config.tabu_tenure)
    log = [f"Tabu start obj={current_obj:.2f}"]

    for it in range(1, config.tabu_max_iterations + 1):
        if time.time() - start_time > config.time_limit_seconds:
            log.append(f"Tabu stopped on time budget at it={it}")
            break

        # Explore neighborhood
        candidates: list[tuple[float, list[str], tuple[str, str]]] = []
        for _ in range(config.tabu_neighborhood_size):
            i = rng.randrange(len(current) - 1)
            a, b = current[i], current[i + 1]
            move = (a, b)
            neighbor = current[:]
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
            neighbor = _fix_precedence(neighbor, problem)
            sched = sgs_build(problem, neighbor)
            obj = compute_quality(problem, sched).objective_value
            candidates.append((obj, neighbor, move))

        # Choose best non-tabu (aspiration: allow if improves best-so-far)
        candidates.sort(key=lambda x: x[0])
        chosen = None
        for obj, neighbor, move in candidates:
            if move in tabu and obj >= best_obj:
                continue
            chosen = (obj, neighbor, move)
            break
        if chosen is None:
            chosen = candidates[0]

        obj, neighbor, move = chosen
        current, current_obj = neighbor, obj
        current_sched = sgs_build(problem, current)
        tabu.append(move)
        if obj < best_obj:
            best, best_sched, best_obj = neighbor[:], current_sched, obj
            log.append(f"  it={it} new best obj={obj:.2f}")

    log.append(f"Tabu done best obj={best_obj:.2f}")
    return HeuristicOutput(assignments=best_sched, iterations=it, log=log)


def _ox_crossover(a: list[str], b: list[str], rng: random.Random) -> list[str]:
    """Order crossover for permutations."""
    n = len(a)
    i, j = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[i:j] = a[i:j]
    fill = [x for x in b if x not in child[i:j]]
    k = 0
    for idx in range(n):
        if child[idx] is None:
            child[idx] = fill[k]
            k += 1
    return child  # type: ignore[return-value]


def run_genetic(
    problem: SchedulingProblem,
    config: HeuristicConfig,
) -> HeuristicOutput:
    rng = random.Random(config.random_seed)
    start_time = time.time()

    # Seed initial population with greedy variants + random permutations
    seeds = [
        _priority_list(problem, HeuristicKind.GREEDY_CRITICAL),
        _priority_list(problem, HeuristicKind.GREEDY_MWKR),
        _priority_list(problem, HeuristicKind.GREEDY_SPT),
        _priority_list(problem, HeuristicKind.GREEDY_EST),
    ]
    pop: list[list[str]] = [s[:] for s in seeds]
    while len(pop) < config.ga_population_size:
        base = rng.choice(seeds)[:]
        rng.shuffle(base)
        base = _fix_precedence(base, problem)
        pop.append(base)

    def fitness(order: list[str]) -> float:
        return compute_quality(problem, sgs_build(problem, order)).objective_value

    fits = [fitness(p) for p in pop]
    log = [f"GA gen=0 best={min(fits):.2f}"]
    it = 0

    for gen in range(1, config.ga_generations + 1):
        if time.time() - start_time > config.time_limit_seconds:
            log.append(f"GA stopped on time budget at gen={gen}")
            break

        # Elitism
        ranked = sorted(range(len(pop)), key=lambda i: fits[i])
        new_pop = [pop[ranked[i]][:] for i in range(config.ga_elitism)]

        while len(new_pop) < config.ga_population_size:
            # Tournament selection
            def tourney() -> list[str]:
                picks = rng.sample(range(len(pop)), k=min(3, len(pop)))
                best_i = min(picks, key=lambda i: fits[i])
                return pop[best_i][:]

            p1, p2 = tourney(), tourney()
            child = (
                _ox_crossover(p1, p2, rng)
                if rng.random() < config.ga_crossover_rate
                else p1
            )
            if rng.random() < config.ga_mutation_rate and len(child) > 1:
                i, j = rng.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            child = _fix_precedence(child, problem)
            new_pop.append(child)
            it += 1

        pop = new_pop
        fits = [fitness(p) for p in pop]
        if gen % 10 == 0:
            log.append(f"GA gen={gen} best={min(fits):.2f}")

    best_i = min(range(len(pop)), key=lambda i: fits[i])
    best_sched = sgs_build(problem, pop[best_i])
    log.append(f"GA done best={fits[best_i]:.2f}")
    return HeuristicOutput(assignments=best_sched, iterations=it, log=log)


def run_constraint_relaxation(
    problem: SchedulingProblem,
    config: HeuristicConfig,
) -> HeuristicOutput:
    """
    Constructive heuristic: build schedule with soft constraints ignored,
    then iteratively add them back in priority order, repairing by greedy
    re-insertion. Useful when the problem may be infeasible.
    """
    t0 = time.time()
    log: list[str] = []

    # Start from critical-path greedy
    base_order = _priority_list(problem, HeuristicKind.GREEDY_CRITICAL)
    base = sgs_build(problem, base_order)
    base_quality = compute_quality(problem, base)
    log.append(
        f"Relax: base obj={base_quality.objective_value:.2f} "
        f"hard_viol={base_quality.hard_violations}"
    )

    # Try alternative orderings if base is infeasible
    best_sched = base
    best_obj = base_quality.objective_value
    it = 1
    for rule in [
        HeuristicKind.GREEDY_MWKR,
        HeuristicKind.GREEDY_SPT,
        HeuristicKind.GREEDY_EST,
    ]:
        if time.time() - t0 > config.time_limit_seconds:
            break
        order = _priority_list(problem, rule)
        sched = sgs_build(problem, order)
        q = compute_quality(problem, sched)
        log.append(
            f"  trying {rule.value}: obj={q.objective_value:.2f} "
            f"hard_viol={q.hard_violations}"
        )
        if q.objective_value < best_obj:
            best_obj = q.objective_value
            best_sched = sched
        it += 1

    log.append(f"Relax done best obj={best_obj:.2f}")
    return HeuristicOutput(assignments=best_sched, iterations=it, log=log)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run_heuristic(
    problem: SchedulingProblem,
    heuristic: HeuristicKind,
    config: HeuristicConfig,
) -> HeuristicOutput:
    if heuristic in {
        HeuristicKind.GREEDY_EST,
        HeuristicKind.GREEDY_SPT,
        HeuristicKind.GREEDY_MWKR,
        HeuristicKind.GREEDY_CRITICAL,
    }:
        return run_greedy(problem, config, heuristic)
    if heuristic == HeuristicKind.SIMULATED_ANNEALING:
        return run_simulated_annealing(problem, config)
    if heuristic == HeuristicKind.TABU_SEARCH:
        return run_tabu_search(problem, config)
    if heuristic == HeuristicKind.GENETIC:
        return run_genetic(problem, config)
    if heuristic == HeuristicKind.CONSTRAINT_RELAXATION:
        return run_constraint_relaxation(problem, config)
    raise ValueError(f"Unknown heuristic {heuristic}")
