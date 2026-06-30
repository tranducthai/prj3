"""
Full experiment: train on planar500, eval on planar500/600 + GGraph.
All runs use llambda=0.7 to match VNS baseline.
"""

import os, json, time
import networkx as nx
from rl import TerritoryDesignProblem, BVNS

MODEL_PATH = "rl_model.pt"
LMBDA      = 0.7
DELTA      = 0.05

PLANAR_DIR  = "Results/MyExperiments"
GGRAPH_DIR  = "Results/MyExperimentsGeneral/27x27Graphs"
os.makedirs(PLANAR_DIR, exist_ok=True)
os.makedirs(GGRAPH_DIR, exist_ok=True)

# delete stale model (was trained with llambda=0.4)
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print(f"Removed old {MODEL_PATH}")

# ─── helpers ────────────────────────────────────────────────────
def make_tdp(G):
    return TerritoryDesignProblem(
        graph_input=G, delta=DELTA, llambda=LMBDA,
        rcl_parameter=0.2, nr_districts=10
    )

def make_bvns(tdp):
    return BVNS(tdp_instance=tdp, shaking_steps=25,
                fail_max=50, nrInitSolutions=50)

def run_instance(file_path, instance_name, save_dir, train=False):
    t0 = time.time()
    G   = nx.read_graphml(file_path)
    tdp = make_tdp(G)
    bvns = make_bvns(tdp)
    bvns.agent.load(MODEL_PATH)
    obj_hist, inf_hist, best_solution, _ = bvns.performBVNS(train_agent=train)
    if train:
        bvns.agent.save(MODEL_PATH)
    elapsed = time.time() - t0

    result = {
        "Instance":      instance_name,
        "Objective":     obj_hist[-1],
        "Infeasibility": inf_hist[-1],
        "Time Taken":    elapsed,
        "nrDistricts":   len(best_solution),
        "delta":         tdp.delta,
        "llambda":       LMBDA,
        "Allocation":    {str(k): [str(n) for n in v]
                          for k, v in best_solution.items()},
    }
    save_path = os.path.join(save_dir, f"{instance_name}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)
    return result, elapsed

# ─── Phase 1: Train on planar500 ────────────────────────────────
print("=" * 60)
print("PHASE 1: TRAINING on planar500  (llambda=0.7)")
print("=" * 60)
for i in range(10):
    name = f"planar500_G{i}"
    fp   = f"TGraphInstances/{name}.graphml"
    print(f"  [{i+1}/10] {name} ...", end=" ", flush=True)
    _, el = run_instance(fp, name, PLANAR_DIR, train=True)
    print(f"done ({el:.0f}s)")
print(f"Phase 1 done. Model -> {MODEL_PATH}\n")

# ─── Phase 2: Eval planar500 + planar600 ────────────────────────
print("=" * 60)
print("PHASE 2: EVALUATING planar500 + planar600")
print("=" * 60)
planar_instances = [
    (f"planar{sz}_G{i}", f"TGraphInstances/planar{sz}_G{i}.graphml")
    for sz in [500, 600] for i in range(10)
    if os.path.exists(f"TGraphInstances/planar{sz}_G{i}.graphml")
]
for idx, (name, fp) in enumerate(planar_instances):
    print(f"  [{idx+1}/{len(planar_instances)}] {name} ...", end=" ", flush=True)
    r, el = run_instance(fp, name, PLANAR_DIR, train=False)
    print(f"Obj={r['Objective']:.3f}  Inf={r['Infeasibility']:.4f}  {el:.0f}s")

# ─── Phase 3: Eval GGraph ────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: EVALUATING GGraph (Center/Corners/Diagonal 486)")
print("=" * 60)
GBASE = "GGraphInstances/newGeneratedInstances/27x27Graphs"
ggraph_instances = [
    (f"{tp}_G{i}", f"{GBASE}/{tp}_G{i}.graphml")
    for tp in ["Center486", "Corners486", "Diagonal486"]
    for i in range(10)
    if os.path.exists(f"{GBASE}/{tp}_G{i}.graphml")
]
for idx, (name, fp) in enumerate(ggraph_instances):
    print(f"  [{idx+1}/{len(ggraph_instances)}] {name} ...", end=" ", flush=True)
    r, el = run_instance(fp, name, GGRAPH_DIR, train=False)
    print(f"Obj={r['Objective']:.3f}  Inf={r['Infeasibility']:.4f}  {el:.0f}s")

# ─── Summary + Comparison ────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: BVNS+RL  vs  VNS")
print(f"{'Instance':<22} {'RL Obj':>8} {'VNS Obj':>8} {'Diff':>7} {'RL Inf':>7} {'VNS Inf':>8}")
print("-" * 70)

def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None

# Planar
for sz in [500, 600]:
    for i in range(10):
        name = f"planar{sz}_G{i}"
        rl  = load_json(f"{PLANAR_DIR}/{name}.json")
        vns = load_json(f"Results/VNSExperiments/{name}_vns.json")
        if rl and vns:
            diff = rl['Objective'] - vns['Objective']
            print(f"{name:<22} {rl['Objective']:>8.3f} {vns['Objective']:>8.3f}"
                  f" {diff:>+7.3f} {rl['Infeasibility']:>7.4f} {vns['Infeasibility']:>8.4f}")

print()
# GGraph
for tp in ["Center486", "Corners486", "Diagonal486"]:
    for i in range(10):
        name_rl  = f"{tp}_G{i}"
        name_vns = f"{tp}_G{i}"
        rl  = load_json(f"{GGRAPH_DIR}/{name_rl}.json")
        vns = load_json(f"Results/VNSExperimentsGeneral/27x27Graphs/{name_vns}_vns.json")
        if rl and vns:
            diff = rl['Objective'] - vns['Objective']
            print(f"{name_rl:<22} {rl['Objective']:>8.3f} {vns['Objective']:>8.3f}"
                  f" {diff:>+7.3f} {rl['Infeasibility']:>7.4f} {vns['Infeasibility']:>8.4f}")

print("\nNegative Diff = RL better, Positive Diff = VNS better")
print("Done.")
