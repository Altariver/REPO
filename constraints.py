"""
Constraint evaluation & penalty computation.

The engine is constraint-kind-agnostic to the heuristics: every heuristic calls
`evaluate()` and gets back a ConstraintReport with hard violation counts and
soft penalty totals. This means new constraint kinds can be added without
touching heuristic code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    Constraint,
    ConstraintKind,
    SchedulingProblem,
    TaskAssignment,
)


@dataclass
class ConstraintReport:
    hard_violations: int = 0
    soft_penalty: float = 0.0
    messages: list[str] = field(default_factory=list)


def _assignment_lookup(
    assignments: list[TaskAssignment],
) -> dict[str, TaskAssignment]:
    return {a.task_id: a for a in assignments}


def _check_pair(a: TaskAssignment, b: TaskAssignment) -> bool:
    """Do two assignments overlap in time?"""
    return a.start < b.end and b.start < a.end


def evaluate(
    problem: SchedulingProblem,
    assignments: list[TaskAssignment],
) -> ConstraintReport:
    """
    Evaluate every constraint (both implicit and user-specified) against the
    given assignments. Returns aggregated report.

    Implicit constraints (always hard):
      - precedence from Task.predecessors
      - deadline from Task.deadline
      - resource capacity from Resource.capacity

    Explicit constraints come from problem.constraints.
    """
    report = ConstraintReport()
    by_id = _assignment_lookup(assignments)
    tasks_by_id = {t.id: t for t in problem.tasks}
    resources_by_id = {r.id: r for r in problem.resources}

    # --- 1. Implicit: precedence -----------------------------------------
    for task in problem.tasks:
        if task.id not in by_id:
            continue
        a = by_id[task.id]
        for pred_id in task.predecessors:
            if pred_id not in by_id:
                report.hard_violations += 1
                report.messages.append(
                    f"Predecessor {pred_id} of {task.id} not scheduled"
                )
                continue
            if by_id[pred_id].end > a.start:
                report.hard_violations += 1
                report.messages.append(
                    f"Precedence violated: {pred_id} ends at "
                    f"{by_id[pred_id].end} but {task.id} starts at {a.start}"
                )

    # --- 2. Implicit: release time & deadline ----------------------------
    for task in problem.tasks:
        if task.id not in by_id:
            continue
        a = by_id[task.id]
        if a.start < task.release_time:
            report.hard_violations += 1
            report.messages.append(
                f"{task.id} starts before release time"
            )
        if task.deadline is not None and a.end > task.deadline:
            report.hard_violations += 1
            report.messages.append(
                f"{task.id} exceeds deadline ({a.end} > {task.deadline})"
            )

    # --- 3. Implicit: resource capacity ----------------------------------
    # For each resource, build a usage profile over time and check capacity.
    horizon = max((a.end for a in assignments), default=0)
    for r_id, resource in resources_by_id.items():
        usage = [0] * (horizon + 1)
        for a in assignments:
            need = a.resources_used.get(r_id, 0)
            if need <= 0:
                # Fall back to task's declared requirements (in case caller
                # did not populate resources_used).
                task = tasks_by_id.get(a.task_id)
                if task is not None:
                    need = task.resource_requirements.get(r_id, 0)
            if need <= 0:
                continue
            for t in range(a.start, a.end):
                if t < len(usage):
                    usage[t] += need
        peak = max(usage) if usage else 0
        if peak > resource.capacity:
            over_units = sum(max(0, u - resource.capacity) for u in usage)
            report.hard_violations += 1
            report.messages.append(
                f"Resource {r_id} over capacity: peak={peak} "
                f"cap={resource.capacity} over_unit_time={over_units}"
            )

    # --- 4. Explicit user constraints -----------------------------------
    for c in problem.constraints:
        violated_amount = _evaluate_single(c, by_id, problem)
        if violated_amount > 0:
            if c.hard:
                report.hard_violations += 1
                report.messages.append(
                    f"Hard constraint {c.id} ({c.kind.value}) violated by "
                    f"{violated_amount}"
                )
            else:
                report.soft_penalty += c.penalty_weight * violated_amount
                report.messages.append(
                    f"Soft constraint {c.id} ({c.kind.value}) violated by "
                    f"{violated_amount}, penalty "
                    f"{c.penalty_weight * violated_amount:.2f}"
                )

    return report


def _evaluate_single(
    c: Constraint,
    by_id: dict[str, TaskAssignment],
    problem: SchedulingProblem,
) -> float:
    """Return violation magnitude (0 means satisfied)."""
    p = c.params
    kind = c.kind

    def get(name: str) -> TaskAssignment | None:
        tid = p.get(name)
        return by_id.get(tid) if tid else None

    if kind == ConstraintKind.PRECEDENCE:
        a, b = get("a"), get("b")
        if a and b and a.end > b.start:
            return a.end - b.start

    elif kind == ConstraintKind.NO_OVERLAP:
        a, b = get("a"), get("b")
        if a and b and _check_pair(a, b):
            return min(a.end, b.end) - max(a.start, b.start)

    elif kind == ConstraintKind.SAME_RESOURCE:
        a, b = get("a"), get("b")
        if a and b:
            shared = set(a.resources_used) & set(b.resources_used)
            return 0 if shared else 1

    elif kind == ConstraintKind.MUTEX_RESOURCE:
        a, b = get("a"), get("b")
        if a and b:
            shared = set(a.resources_used) & set(b.resources_used)
            return len(shared)

    elif kind == ConstraintKind.TIME_WINDOW:
        a = get("a")
        earliest = p.get("earliest", 0)
        latest = p.get("latest")
        if a:
            viol = 0
            if a.start < earliest:
                viol += earliest - a.start
            if latest is not None and a.end > latest:
                viol += a.end - latest
            return viol

    elif kind == ConstraintKind.MIN_GAP:
        a, b = get("a"), get("b")
        gap = p.get("gap", 0)
        if a and b:
            actual = b.start - a.end
            if actual < gap:
                return gap - actual

    elif kind == ConstraintKind.MAX_GAP:
        a, b = get("a"), get("b")
        gap = p.get("gap", 0)
        if a and b:
            actual = b.start - a.end
            if actual > gap:
                return actual - gap

    elif kind == ConstraintKind.RESOURCE_CAP:
        # Override cap in window - computed against assignments
        r_id = p.get("resource_id")
        window_start = p.get("start", 0)
        window_end = p.get("end", 10**9)
        override_cap = p.get("capacity", 0)
        if r_id is None:
            return 0
        peak = 0
        for t in range(window_start, window_end):
            usage = 0
            for a in by_id.values():
                if a.start <= t < a.end:
                    usage += a.resources_used.get(r_id, 0)
            peak = max(peak, usage)
        if peak > override_cap:
            return peak - override_cap

    elif kind == ConstraintKind.CUSTOM:
        # Custom constraints are opaque. We expect users to pre-compute
        # and pass a `violation` magnitude directly.
        return float(p.get("violation", 0))

    return 0
