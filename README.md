## What this is

This repository implements **ALNS+RL Repair**, a hybrid metaheuristic combining Adaptive Large Neighbourhood Search (ALNS) with Reinforcement Learning (Q-learning) to solve the **Delivery Territory Design Problem (DTDP)**. The DTDP partitions geographic units (graph nodes) into balanced delivery territories while minimizing geographic dispersion. The approach replaces traditional greedy repair strategies with a learned RL policy that makes node-to-territory assignment decisions.

### Stack
- **Language(s):** Python
- **Framework / runtime:** NumPy, NetworkX (graph processing)
- **Notable libraries:** SciPy (statistics), scikit-learn compatibility

## How it's organized

```
.
├── dtdp.py                      # Core DTDP problem class: graph handling, constraint checking,
│                                 # local search, constructive heuristics
├── alns_rl_repair.py            # ALNS framework with RL repair agent, destroy operators,
│                                 # state/reward/Q-learning mechanics, experimental pipeline
├── experiment.ipynb             # Jupyter notebook: results visualization, ablation studies
├── report.tex                   # Full research report (Vietnamese): problem formulation,
│                                 # algorithm details, extensive experimental analysis
├── model_tgraph.npz             # Pre-trained Q-learning weights for random planar graphs
├── model_ggraph.npz             # Pre-trained Q-learning weights for grid graphs
├── TGraphInstances/             # Benchmark: 30 random planar graphs (500/600/700 nodes)
├── GGraphInstances/             # Benchmark: 90 grid graphs (3 sizes × 3 depot configs × 10 instances)
└── Results/                     # Detailed experimental results and comparisons
```

**How it fits together:** The workflow initializes territory designs greedily (selecting dispersed centroids, then assigning nodes). Each ALNS iteration destroys k nodes from a chosen territory (random/worst-diameter/infeasible strategy), then the RL agent repairs by iteratively assigning each removed node to the best territory according to a learned Q-function. The Q-function encodes 8 features (workload balance, diameter, geometry) and is updated via linear TD(0) Q-learning. A Simulated Annealing criterion gates acceptance of repair outcomes.

## How to run it

**Install dependencies:**
```bash
pip install numpy networkx scipy
```

**Train and evaluate on a single instance:**
```python
import networkx as nx
from dtdp import TerritoryDesignProblem
from alns_rl_repair import alns_rl_repair, alns_greedy_repair

# Load or create a graph
G = nx.read_graphml('TGraphInstances/planar500_G0.graphml')

# Create DTDP instance
tdp = TerritoryDesignProblem(G, delta=0.05, llambda=0.7, 
                              rcl_parameter=0.2, nr_districts=10)

# Run ALNS+RL Repair (trains a new agent)
obj, inf, solution, history, agent = alns_rl_repair(
    tdp, n_iterations=300, k_remove=10, lam=0.7, seed=42, train=True
)

print(f"Objective (max diameter): {obj}")
print(f"Infeasibility (balance violation): {inf}")

# Save trained agent
agent.save('my_model.npz')
```

**Run full benchmark comparison (requires graph files):**
```bash
python alns_rl_repair.py
```
Produces comparison tables: ALNS+RL Repair vs. VNS baseline on TGraph (planar) and GGraph (grid) instances, with statistical significance testing.

**Evaluate with pre-trained model:**
```python
from alns_rl_repair import RepairAgent

agent = RepairAgent(n_features=8, n_actions=10)
agent.load('model_ggraph.npz')
agent.epsilon = 0.05  # near-greedy mode

obj, inf, solution, history, _ = alns_rl_repair(
    tdp, agent=agent, n_iterations=300, train=False
)
```

## Try asking

- **How does the RL repair choose which territory to assign a removed node to, and what state features influence that decision?** (See `build_state()` and `RepairAgent.select_action()` in `alns_rl_repair.py` for the 8-dimensional feature engineering and ε-greedy selection.)

- **Why does the method achieve near-perfect load balance (Inf ≈ 0) on grid graphs but struggle on planar graphs?** (The report's discussion traces this to: grid structure enables the agent to learn a simple "assign node near territory" policy that generalizes; planar graph heterogeneity defeats node-by-node greedy repair, requiring global lookahead.)

- **How does the algorithm compare to Variable Neighbourhood Search (VNS), and on which instance types does it win?** (Report shows: GGraph Center486/600 within 6–9% of VNS while maintaining feasibility; TGraph underperforms VNS by ~18%. Some individual instances beat VNS due to learned policy exploiting geometry better.)
