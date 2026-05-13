"""
End-to-end test client for the Scheduling Optimization API.

Usage:
    # 1. Start the API in another terminal:
    uvicorn scheduling_api.main:app --reload

    # 2. Run this client:
    python -m scheduling_api.test_client

It exercises:
    - Problem creation
    - Constraint attachment
    - Recommendation
    - Solve with multiple heuristics
    - Multi-scenario simulation
    - Results retrieval & ranking
"""
from __future__ import annotations

import json
from pprint import pprint

import httpx

BASE = "http://localhost:8000/api/v1"


def build_problem() -> dict:
    """A medium-sized RCPSP instance inspired by production scheduling."""
    tasks = []
    # Three parallel "recipes" each with 4 stages
    for recipe in range(3):
        prefix = f"R{recipe}"
        # Preparation (needs workers)
        tasks.append({
            "id": f"{prefix}_prep",
            "name": f"Recipe {recipe} prep",
            "duration": 3,
            "resource_requirements": {"workers": 2},
            "predecessors": [],
            "release_time": 0,
            "weight": 1.0,
        })
        # Mixing (needs a mixer + workers)
        tasks.append({
            "id": f"{prefix}_mix",
            "duration": 4,
            "resource_requirements": {"mixer": 1, "workers": 1},
            "predecessors": [f"{prefix}_prep"],
        })
        # Bake (needs oven)
        tasks.append({
            "id": f"{prefix}_bake",
            "duration": 5,
            "resource_requirements": {"oven": 1},
            "predecessors": [f"{prefix}_mix"],
            "due_date": 15,
            "weight": 2.0,
        })
        # Pack (needs workers)
        tasks.append({
            "id": f"{prefix}_pack",
            "duration": 2,
            "resource_requirements": {"workers": 2},
            "predecessors": [f"{prefix}_bake"],
            "deadline": 20,
        })

    resources = [
        {"id": "workers", "capacity": 4, "cost_per_unit_time": 20.0},
        {"id": "mixer", "capacity": 2, "cost_per_unit_time": 5.0},
        {"id": "oven", "capacity": 2, "cost_per_unit_time": 15.0},
    ]

    return {
        "name": "Bakery Production",
        "description": "3 recipes x 4 stages with shared resources",
        "resources": resources,
        "tasks": tasks,
        "constraints": [],
        "objectives": [
            {"kind": "makespan", "weight": 1.0},
            {"kind": "weighted_tardiness", "weight": 2.0},
        ],
    }


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        # --- Health -------------------------------------------------------
        print("== 1. Health check ==")
        r = c.get("/health")
        r.raise_for_status()
        print(r.json())

        # --- Create problem ----------------------------------------------
        print("\n== 2. Create problem ==")
        r = c.post("/problems", json=build_problem())
        r.raise_for_status()
        problem = r.json()
        pid = problem["id"]
        print(f"Created problem {pid} with "
              f"{len(problem['tasks'])} tasks, "
              f"{len(problem['resources'])} resources")

        # --- Add a soft constraint ---------------------------------------
        print("\n== 3. Add soft constraint (min gap between R0_bake and R1_bake) ==")
        constraint = {
            "kind": "min_gap",
            "hard": False,
            "priority": 50,
            "penalty_weight": 3.0,
            "params": {"a": "R0_bake", "b": "R1_bake", "gap": 2},
            "description": "Ovens need 2 time units cool-down between bakes",
        }
        r = c.post(f"/problems/{pid}/constraints", json=constraint)
        r.raise_for_status()
        print(f"Problem now has {len(r.json()['constraints'])} constraints")

        # --- Recommend ----------------------------------------------------
        print("\n== 4. Get heuristic recommendations ==")
        r = c.post("/recommend", json={
            "problem_id": pid,
            "time_budget_seconds": 5.0,
            "primary_objective": "makespan",
            "prefer": "balanced",
        })
        r.raise_for_status()
        rec = r.json()
        print("Problem features:")
        pprint(rec["problem_features"])
        print("\nTop 3 recommendations:")
        for i, rr in enumerate(rec["ranked_recommendations"][:3], 1):
            print(f"  {i}. {rr['heuristic']:25s} "
                  f"confidence={rr['confidence']:.2f}")
            for line in rr["rationale"][:2]:
                print(f"        - {line}")

        # --- Single solve with greedy ------------------------------------
        print("\n== 5. Solve with greedy_critical ==")
        r = c.post("/solve", json={
            "problem_id": pid,
            "heuristic": "greedy_critical",
            "config": {"time_limit_seconds": 2.0},
        })
        r.raise_for_status()
        sol = r.json()
        q = sol["quality"]
        print(f"Objective={q['objective_value']:.1f}, "
              f"Makespan={q['makespan']}, "
              f"Feasible={q['feasible']}, "
              f"Tardiness={q['weighted_tardiness']:.1f}, "
              f"Hard violations={q['hard_violations']}")

        # --- Simulation: compare heuristics and scenarios ----------------
        print("\n== 6. Multi-heuristic multi-scenario simulation ==")
        sim_request = {
            "problem_id": pid,
            "heuristics": [
                "greedy_critical",
                "greedy_mwkr",
                "simulated_annealing",
                "tabu_search",
                "genetic",
            ],
            "scenarios": [
                {"name": "baseline"},
                {
                    "name": "broken_oven",
                    "resource_capacity_changes": {"oven": 1},
                },
                {
                    "name": "rush_job",
                    "task_duration_multipliers": {
                        "R0_bake": 0.6,
                        "R1_bake": 0.6,
                    },
                },
                {
                    "name": "staff_shortage",
                    "resource_capacity_changes": {"workers": 2},
                },
            ],
            "config": {
                "time_limit_seconds": 2.0,
                "random_seed": 42,
            },
        }
        r = c.post("/simulate", json=sim_request)
        r.raise_for_status()
        sim = r.json()

        print(f"Total wall time: {sim['total_wall_time_seconds']:.2f}s")
        print("\nScenario summary:")
        for s in sim["scenarios"]:
            print(f"\n  Scenario: {s['scenario_name']}")
            print(f"  {'Heuristic':<25} {'Obj':>8} {'Makespan':>9} "
                  f"{'Feasible':>9} {'Time(s)':>8}")
            for so in s["solutions"]:
                print(f"  {so['heuristic']:<25} "
                      f"{so['quality']['objective_value']:>8.1f} "
                      f"{so['quality']['makespan']:>9d} "
                      f"{str(so['quality']['feasible']):>9} "
                      f"{so['wall_time_seconds']:>8.2f}")

        print("\nCross-scenario summary:")
        pprint(sim["cross_scenario_summary"])

        # --- List all solutions for the problem ranked -------------------
        print("\n== 7. All solutions (ranked best-first) ==")
        r = c.get(f"/problems/{pid}/solutions")
        r.raise_for_status()
        ranked = r.json()
        print(f"Total solutions: {len(ranked['solutions'])}, "
              f"best objective: {ranked['best_objective']:.1f}")
        print("\nTop 5:")
        for s in ranked["solutions"][:5]:
            print(f"  {s['heuristic']:<25} "
                  f"obj={s['quality']['objective_value']:.1f} "
                  f"problem={s['problem_id']}")

        print("\n== 8. Done - API spec available at http://localhost:8000/docs ==")


if __name__ == "__main__":
    main()
