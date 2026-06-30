"""
Run BVNS+RL on TGraph instances (planar500, planar600).
Phase 1: Train agent on planar500 (saves/loads rl_model.pt after each run).
Phase 2: Evaluate (no training) on planar500 + planar600, save JSON results.
"""

import os
import json
import time
import networkx as nx
from rl import TerritoryDesignProblem, BVNS

MODEL_PATH = "rl_model.pt"
RESULTS_DIR = "Results/MyExperiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Phase 1: Train ──────────────────────────────────────────────
print("=" * 55)
print("PHASE 1: TRAINING on planar500 (10 instances)")
print("=" * 55)

for i in range(10):
    instance_name = f"planar500_G{i}"
    file_path = f"TGraphInstances/{instance_name}.graphml"
    t0 = time.time()
    print(f"  [{i+1}/10] {instance_name} ...", end=" ", flush=True)

    G = nx.read_graphml(file_path)
    tdp = TerritoryDesignProblem(
        graph_input=G, delta=0.05, llambda=0.4,
        rcl_parameter=0.2, nr_districts=10
    )
    bvns = BVNS(
        tdp_instance=tdp, shaking_steps=25,
        fail_max=50, nrInitSolutions=50
    )
    bvns.agent.load(MODEL_PATH)
    bvns.performBVNS(train_agent=True)
    bvns.agent.save(MODEL_PATH)
    print(f"done ({time.time()-t0:.1f}s)")

print(f"\nPhase 1 complete. Model saved -> {MODEL_PATH}\n")

# ─── Phase 2: Evaluate ───────────────────────────────────────────
print("=" * 55)
print("PHASE 2: EVALUATING planar500 + planar600")
print("=" * 55)

sizes = [500, 600]
total = sum(
    1 for size in sizes for i in range(10)
    if os.path.exists(f"TGraphInstances/planar{size}_G{i}.graphml")
)
done = 0

for size in sizes:
    for i in range(10):
        instance_name = f"planar{size}_G{i}"
        file_path = f"TGraphInstances/{instance_name}.graphml"

        if not os.path.exists(file_path):
            print(f"  [SKIP] {instance_name}")
            continue

        done += 1
        print(f"  [{done}/{total}] {instance_name} ...", end=" ", flush=True)
        t0 = time.time()

        G = nx.read_graphml(file_path)
        tdp = TerritoryDesignProblem(
            graph_input=G, delta=0.05, llambda=0.4,
            rcl_parameter=0.2, nr_districts=10
        )
        bvns = BVNS(
            tdp_instance=tdp, shaking_steps=25,
            fail_max=50, nrInitSolutions=50
        )
        bvns.agent.load(MODEL_PATH)
        obj_hist, inf_hist, best_solution, timeline = bvns.performBVNS(train_agent=False)
        elapsed = time.time() - t0

        result = {
            "Instance": instance_name,
            "Objective": obj_hist[-1],
            "Infeasibility": inf_hist[-1],
            "Time Taken": elapsed,
            "nrDistricts": len(best_solution),
            "delta": tdp.delta,
            "llambda": tdp.llambda,
            "Allocation": {
                str(k): [str(n) for n in nodes]
                for k, nodes in best_solution.items()
            },
        }

        save_path = os.path.join(RESULTS_DIR, f"{instance_name}.json")
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)

        print(
            f"Obj={result['Objective']:.3f}  "
            f"Inf={result['Infeasibility']:.4f}  "
            f"Time={elapsed:.1f}s  → {save_path}"
        )

# ─── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'Instance':<20} {'Objective':>10} {'Infeasibility':>14} {'Time(s)':>8}")
print("-" * 55)
for size in sizes:
    for i in range(10):
        name = f"planar{size}_G{i}"
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            print(
                f"{d['Instance']:<20} {d['Objective']:>10.3f} "
                f"{d['Infeasibility']:>14.4f} {d['Time Taken']:>8.1f}"
            )
