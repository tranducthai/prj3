"""
ALNS + RL Repair for DTDP
RL được dùng trong REPAIR phase:
  - Mỗi bước: gán 1 node vào 1 territory
  - State: features của node + territory được xét
  - Action: territory index
  - Reward: cải thiện merit sau mỗi lần gán
"""

import os
import networkx as nx
import numpy as np
import random
import copy
import time
from scipy import stats

activities = ["workload", "n_customers", "demand"]

# ─────────────────────────────────────────────
# DESTROY OPERATORS
# ─────────────────────────────────────────────

def destroy_random(solution, district_act, tdp, k):
    removed, sol, act = [], copy.deepcopy(solution), copy.deepcopy(district_act)
    all_nodes = [(n,d) for d,nodes in sol.items() for n in nodes]
    for node, dist in random.sample(all_nodes, min(k, len(all_nodes))):
        sol[dist].remove(node)
        for a in activities: act[dist][a] -= tdp.graph_input.nodes[node][a]
        removed.append(node)
    return sol, act, removed

def destroy_worst_diameter(solution, district_act, tdp, k):
    removed, sol, act = [], copy.deepcopy(solution), copy.deepcopy(district_act)
    diams = {d: tdp.get_district_diameter_numpy(sol[d]) for d in sol if sol[d]}
    worst = max(diams, key=lambda d: diams[d]['diameter'])
    dn = list(diams[worst]['diameter_nodes'])
    cands = list(set(dn + [n for w in dn for n in tdp.graph_input.adj[w] if n in sol[worst]]))[:k]
    for node in cands:
        if node in sol[worst]:
            sol[worst].remove(node)
            for a in activities: act[worst][a] -= tdp.graph_input.nodes[node][a]
            removed.append(node)
    return sol, act, removed

def destroy_infeasible(solution, district_act, tdp, k):
    removed, sol, act = [], copy.deepcopy(solution), copy.deepcopy(district_act)
    inf_per = {d: tdp.get_district_infeasibility(sol[d], act[d])[0] for d in sol if sol[d]}
    worst = max(inf_per, key=inf_per.get)
    if inf_per[worst] == 0:
        return destroy_random(solution, district_act, tdp, k)
    for node in random.sample(sol[worst], min(k, len(sol[worst]))):
        sol[worst].remove(node)
        for a in activities: act[worst][a] -= tdp.graph_input.nodes[node][a]
        removed.append(node)
    return sol, act, removed

DESTROY_OPS = [destroy_random, destroy_worst_diameter, destroy_infeasible]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def compute_FG(solution, district_act, tdp):
    F = max(tdp.get_district_diameter_numpy(solution[d])['diameter']
            for d in solution if solution[d])
    G = sum(tdp.get_district_infeasibility(solution[d], district_act[d])[0]
            for d in solution)
    return F, G

def merit(F, G, tdp, lam):
    return lam*(F/tdp.graph_diameter) + (1-lam)*G

def sa_accept(old, new, T):
    return new < old or random.random() < np.exp(-(new-old)/max(T,1e-10))

def build_initial(tdp, lam, tries=5):
    best_m, best_sol, best_act = float('inf'), None, None
    for _ in range(tries):
        tdp.centers_depots = tdp.select_centroids()
        _, _, sol, act = tdp.constructDistricts()
        F, G = compute_FG(sol, act, tdp)
        m = merit(F, G, tdp, lam)
        if m < best_m:
            best_m, best_sol, best_act = m, copy.deepcopy(sol), copy.deepcopy(act)
    return best_sol, best_act

# ─────────────────────────────────────────────
# STATE BUILDER
# ─────────────────────────────────────────────

def build_state(node, district_id, solution, district_act, tdp, mu):
    """
    Feature vector mô tả việc gán 'node' vào 'district_id'.
    Shape: (len(activities)*2 + 2,) = 8 features
    
    Features:
      [0..2] excess của district (normalized): balance hiện tại
      [3]    diameter district (normalized)
      [4..6] attributes của node (normalized)
      [7]    mean distance node đến district (normalized)
    """
    excess = np.array([
        (district_act[district_id][a] - mu[a]) / max(mu[a], 1e-6)
        for a in activities
    ])
    diam = tdp.get_district_diameter_numpy(solution[district_id])['diameter']
    diam_norm = diam / tdp.graph_diameter
    node_attrs = np.array([
        tdp.graph_input.nodes[node][a] / max(mu[a], 1e-6)
        for a in activities
    ])
    if solution[district_id]:
        # dùng shortest_paths_arr thay vì dict để nhanh hơn
        node_idx = tdp.nodes_index[node]
        dist_idxs = [tdp.nodes_index[n2] for n2 in solution[district_id]]
        mean_dist = np.mean(tdp.shortest_paths_arr[node_idx][dist_idxs]) / tdp.graph_diameter
    else:
        mean_dist = 0.0

    return np.concatenate([excess, [diam_norm], node_attrs, [mean_dist]])

# ─────────────────────────────────────────────
# RL AGENT — Linear Q-learning
# ─────────────────────────────────────────────

class RepairAgent:
    """
    Linear Q-learning: Q(s,a) = w_a^T · s
    
    Mỗi action (territory) có 1 weight vector riêng.
    State là feature vector của (node, territory) pair.
    
    Tại sao linear thay vì Q-table?
    → State là continuous vector, không thể dùng Q-table trực tiếp.
    → Linear đủ đơn giản để học nhanh, đủ expressive để capture
      relationship giữa features và Q-value.
    """
    def __init__(self, n_features, n_actions,
                 alpha=0.01, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.n_features    = n_features
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        # W[a] = weight vector cho action a, shape (n_features,)
        self.W = np.zeros((n_actions, n_features))

    def q(self, state, action):
        return float(self.W[action] @ state)

    def select_action(self, states_per_action):
        """
        states_per_action: list of state vectors, 1 per action.
        Tính Q(s_a, a) cho mỗi action a, chọn argmax (hoặc random).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_vals = np.array([self.q(states_per_action[a], a)
                           for a in range(self.n_actions)])
        return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_states, done):
        if done:
            td_target = reward
        else:
            next_q = max(self.q(next_states[a], a) for a in range(self.n_actions))
            td_target = reward + self.gamma * next_q
        td_error = td_target - self.q(state, action)
        self.W[action] += self.alpha * td_error * state

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.savez(path, W=self.W, epsilon=np.array([self.epsilon]))

    def load(self, path):
        if os.path.exists(path):
            data = np.load(path)
            self.W       = data['W']
            self.epsilon = float(data['epsilon'][0])
            return True
        return False

# ─────────────────────────────────────────────
# RL REPAIR
# ─────────────────────────────────────────────

def repair_rl(solution, district_act, removed_nodes, tdp, lam, agent, mu, train=True):
    sol, act = copy.deepcopy(solution), copy.deepcopy(district_act)
    districts = list(sol.keys())

    # gán node có tổng attribute lớn trước
    sorted_nodes = sorted(
        removed_nodes,
        key=lambda v: sum(tdp.graph_input.nodes[v][a] for a in activities),
        reverse=True
    )

    F_cur, G_cur = compute_FG(sol, act, tdp)
    m_cur = merit(F_cur, G_cur, tdp, lam)

    transitions = []

    for idx, node in enumerate(sorted_nodes):
        done = (idx == len(sorted_nodes) - 1)

        # build state cho mỗi territory
        states_per_action = [
            build_state(node, d, sol, act, tdp, mu)
            for d in districts
        ]

        # chọn territory
        action = agent.select_action(states_per_action)
        chosen_dist = districts[action]
        state = states_per_action[action]

        # thực hiện gán
        sol[chosen_dist].append(node)
        for a in activities:
            act[chosen_dist][a] += tdp.graph_input.nodes[node][a]

        # reward = cải thiện merit ngay lập tức
        F_new, G_new = compute_FG(sol, act, tdp)
        m_new = merit(F_new, G_new, tdp, lam)
        reward = m_cur - m_new   # dương nếu cải thiện

        # next states
        if not done and train:
            next_node = sorted_nodes[idx+1]
            next_states = [
                build_state(next_node, d, sol, act, tdp, mu)
                for d in districts
            ]
        else:
            next_states = states_per_action  # terminal

        if train:
            agent.update(state, action, reward, next_states, done)

        transitions.append((state, action, reward))
        m_cur = m_new

    return sol, act, transitions

def repair_greedy(solution, district_act, removed_nodes, tdp, lam):
    sol, act = copy.deepcopy(solution), copy.deepcopy(district_act)
    sorted_nodes = sorted(
        removed_nodes,
        key=lambda v: sum(tdp.graph_input.nodes[v][a] for a in activities),
        reverse=True
    )
    cur_diams = {d: tdp.get_district_diameter_numpy(sol[d]) for d in sol}
    cur_inf   = {d: tdp.get_district_infeasibility(sol[d], act[d])[0] for d in sol}
    for node in sorted_nodes:
        best_m, best_d = float('inf'), None
        for d in sol:
            td = tdp.get_district_diameter_numpy(sol[d], node)['diameter']
            nd = max(td, max(cur_diams[dd]['diameter'] for dd in sol if dd != d))
            na = {a: act[d][a] + tdp.graph_input.nodes[node][a] for a in activities}
            ni = tdp.get_district_infeasibility(sol[d], na)[0]
            ni_total = sum(cur_inf[dd] for dd in sol if dd != d) + ni
            m = lam*(nd/tdp.graph_diameter) + (1-lam)*ni_total
            if m < best_m: best_m, best_d = m, d
        sol[best_d].append(node)
        for a in activities: act[best_d][a] += tdp.graph_input.nodes[node][a]
        cur_diams[best_d] = tdp.get_district_diameter_numpy(sol[best_d])
        cur_inf[best_d]   = tdp.get_district_infeasibility(sol[best_d], act[best_d])[0]
    return sol, act

# ─────────────────────────────────────────────
# ALNS + RL REPAIR
# ─────────────────────────────────────────────

def alns_rl_repair(tdp, agent=None, n_iterations=300, k_remove=10,
                   T_init=0.05, T_decay=0.995, lam=0.4, seed=42, train=True):
    random.seed(seed); np.random.seed(seed)

    current_sol, current_act = build_initial(tdp, lam)
    current_F, current_G = compute_FG(current_sol, current_act, tdp)
    current_m = merit(current_F, current_G, tdp, lam)

    best_sol = copy.deepcopy(current_sol)
    best_F, best_G, best_m = current_F, current_G, current_m

    mu = tdp.totalAverageAct  # precomputed by TerritoryDesignProblem

    n_features = len(activities)*2 + 2
    if agent is None:
        agent = RepairAgent(n_features=n_features, n_actions=tdp.nr_districts,
                            alpha=0.01, gamma=0.9, epsilon=1.0,
                            epsilon_min=0.05, epsilon_decay=0.995)

    T = T_init
    history = {'obj': [], 'inf': [], 'reward': []}

    for _ in range(n_iterations):
        op = random.choice(DESTROY_OPS)
        new_sol, new_act, removed = op(current_sol, current_act, tdp, k_remove)
        new_sol, new_act, transitions = repair_rl(
            new_sol, new_act, removed, tdp, lam, agent, mu, train=train)
        if train:
            agent.decay_epsilon()

        new_F, new_G = compute_FG(new_sol, new_act, tdp)
        new_m = merit(new_F, new_G, tdp, lam)

        if sa_accept(current_m, new_m, T):
            current_sol, current_act = new_sol, new_act
            current_F, current_G, current_m = new_F, new_G, new_m

        if new_m < best_m:
            best_sol = copy.deepcopy(new_sol)
            best_F, best_G, best_m = new_F, new_G, new_m

        T *= T_decay
        history['obj'].append(best_F)
        history['inf'].append(best_G)
        history['reward'].append(sum(t[2] for t in transitions))

    return best_F, best_G, best_sol, history, agent


def alns_greedy_repair(tdp, n_iterations=300, k_remove=10,
                       T_init=0.05, T_decay=0.995, lam=0.4, seed=42):
    random.seed(seed); np.random.seed(seed)

    current_sol, current_act = build_initial(tdp, lam)
    current_F, current_G = compute_FG(current_sol, current_act, tdp)
    current_m = merit(current_F, current_G, tdp, lam)

    best_sol = copy.deepcopy(current_sol)
    best_F, best_G, best_m = current_F, current_G, current_m

    T = T_init
    history = {'obj': [], 'inf': []}

    for _ in range(n_iterations):
        op = random.choice(DESTROY_OPS)
        new_sol, new_act, removed = op(current_sol, current_act, tdp, k_remove)
        new_sol, new_act = repair_greedy(new_sol, new_act, removed, tdp, lam)

        new_F, new_G = compute_FG(new_sol, new_act, tdp)
        new_m = merit(new_F, new_G, tdp, lam)

        if sa_accept(current_m, new_m, T):
            current_sol, current_act = new_sol, new_act
            current_F, current_G, current_m = new_F, new_G, new_m

        if new_m < best_m:
            best_sol = copy.deepcopy(new_sol)
            best_F, best_G, best_m = new_F, new_G, new_m

        T *= T_decay
        history['obj'].append(best_F)
        history['inf'].append(best_G)

    return best_F, best_G, best_sol, history

# ─────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────

def run_experiment(graph_path, n_iterations=300, k_remove=10,
                   lam=0.4, delta=0.05, n_runs=10):
    from dtdp import TerritoryDesignProblem
    G = nx.read_graphml(graph_path)
    rl_obj, rl_inf, rl_time = [], [], []
    gr_obj, gr_inf, gr_time = [], [], []

    for run in range(n_runs):
        seed = run * 17

        tdp = TerritoryDesignProblem(G, delta=delta, llambda=lam,
                                      rcl_parameter=0.2, nr_districts=10)
        t0 = time.time()
        obj, inf = alns_rl_repair(tdp, n_iterations=n_iterations,
                                   k_remove=k_remove, lam=lam, seed=seed)[:2]
        rl_obj.append(obj); rl_inf.append(inf); rl_time.append(time.time()-t0)

        tdp = TerritoryDesignProblem(G, delta=delta, llambda=lam,
                                      rcl_parameter=0.2, nr_districts=10)
        t0 = time.time()
        obj, inf = alns_greedy_repair(tdp, n_iterations=n_iterations,
                                       k_remove=k_remove, lam=lam, seed=seed)[:2]
        gr_obj.append(obj); gr_inf.append(inf); gr_time.append(time.time()-t0)

    t_obj, p_obj = stats.ttest_rel(rl_obj, gr_obj)
    t_inf, p_inf = stats.ttest_rel(rl_inf, gr_inf)

    return {
        'rl': {'obj': rl_obj, 'inf': rl_inf, 'time': rl_time},
        'gr': {'obj': gr_obj, 'inf': gr_inf, 'time': gr_time},
        'stats': {'p_obj': p_obj, 'p_inf': p_inf,
                  't_obj': t_obj, 't_inf': t_inf}
    }


def print_result(res, name):
    rl, gr, st = res['rl'], res['gr'], res['stats']
    print(f"\n{'='*65}")
    print(f"Instance: {name}  (n_runs={len(rl['obj'])})")
    print(f"{'='*65}")
    print(f"{'Metric':<22} {'ALNS+RL Repair':>18} {'ALNS+Greedy':>18}  p-value")
    print(f"{'-'*65}")

    def fmt(v): return f"{np.mean(v):>7.2f}±{np.std(v):<7.2f}"
    print(f"{'Objective (F)':<22} {fmt(rl['obj']):>18} {fmt(gr['obj']):>18}"
          f"  {st['p_obj']:.3f}{'*' if st['p_obj']<0.05 else ' '}")
    print(f"{'Infeasibility (G)':<22} {fmt(rl['inf']):>18} {fmt(gr['inf']):>18}"
          f"  {st['p_inf']:.3f}{'*' if st['p_inf']<0.05 else ' '}")
    print(f"{'Time (s)':<22} {np.mean(rl['time']):>7.1f}{'':>11}"
          f"{np.mean(gr['time']):>7.1f}")
    print(f"\n* = statistically significant (p < 0.05)")
    print(f"\nInterpretation:")
    for metric, p, rv, gv in [('Objective',     st['p_obj'], rl['obj'], gr['obj']),
                                ('Infeasibility', st['p_inf'], rl['inf'], gr['inf'])]:
        if p < 0.05:
            d = np.mean(gv) - np.mean(rv)
            print(f"  {metric}: ALNS+RL {'BETTER' if d>0 else 'WORSE'} "
                  f"({abs(d):.3f} units, p={p:.3f})")
        else:
            print(f"  {metric}: No significant difference (p={p:.3f})")


def analyze_weights(agent, name):
    """Phân tích weight vectors đã học — thay cho Q-table."""
    print(f"\n{'='*65}")
    print(f"LEARNED WEIGHTS — {name}")
    print(f"{'='*65}")
    feat_names = ['excess_demand', 'excess_workload', 'excess_customers',
                  'diameter_norm', 'node_demand', 'node_workload',
                  'node_customers', 'mean_dist']
    # mean weight across all actions
    mean_w = np.mean(agent.W, axis=0)
    print(f"\nMean weight per feature (averaged across territories):")
    for fname, w in zip(feat_names, mean_w):
        bar = '█' * int(abs(w)*50) if abs(w) < 2 else '█'*10
        sign = '+' if w >= 0 else '-'
        print(f"  {fname:<20}: {sign}{abs(w):.4f}  {bar}")
    print(f"\nInterpretation:")
    print(f"  Negative weight on 'excess_*' → avoid over-loaded territories ✓")
    print(f"  Negative weight on 'diameter_norm' → avoid large-diameter territories ✓")
    print(f"  Negative weight on 'mean_dist' → prefer nearby territories ✓")


if __name__ == '__main__':
    from dtdp import TerritoryDesignProblem
    base = os.path.dirname(os.path.abspath(__file__))

    instances = {
        'T-500':   'TGraphInstances/planar500_G0.graphml',
        'T-600':   'TGraphInstances/planar600_G0.graphml',
        'G-C486':  'GGraphInstances/newGeneratedInstances/27x27Graphs/Center486_G0.graphml',
        'G-D486':  'GGraphInstances/newGeneratedInstances/27x27Graphs/Diagonal486_G0.graphml',
        'G-CN486': 'GGraphInstances/newGeneratedInstances/27x27Graphs/Corners486_G0.graphml',
    }

    print("ALNS + RL Repair vs ALNS + Greedy Repair on DTDP")
    print("n_iterations=300, k_remove=10, n_runs=10\n")

    last_agent, last_name = None, None
    for name, rel in instances.items():
        full = os.path.join(base, rel)
        if not os.path.exists(full):
            print(f"[SKIP] {name}"); continue
        print(f"Running {name}...", end=' ', flush=True)
        res = run_experiment(full, n_iterations=300, k_remove=10, n_runs=10)
        print("done")
        print_result(res, name)

        G = nx.read_graphml(full)
        tdp = TerritoryDesignProblem(G, delta=0.05, llambda=0.4,
                                      rcl_parameter=0.2, nr_districts=10)
        _, _, _, _, last_agent = alns_rl_repair(tdp, n_iterations=300,
                                                 k_remove=10, lam=0.4, seed=0)
        last_name = name

    if last_agent:
        analyze_weights(last_agent, last_name)
