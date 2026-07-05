import networkx as nx
import numpy as np
import random
import copy as cp

activities = ["workload", "n_customers", "demand"]


class TerritoryDesignProblem:
    def __init__(self, graph_input, delta, llambda, rcl_parameter, nr_districts=10):
        self.graph_input = graph_input
        self.delta = delta
        self.llambda = llambda
        self.rcl_parameter = rcl_parameter
        self.nodes = list(graph_input.nodes())
        getFurthestNodesResponse = self.get_furthest_nodes()
        self.shortest_paths_dict = getFurthestNodesResponse["node_shortest_path_dicts"]
        self.graph_diameter = getFurthestNodesResponse["diameter"]
        self.nodes_index = {v: i for i, v in enumerate(self.graph_input.nodes)}
        self.shortest_paths_arr = np.zeros(
            (len(self.nodes_index), len(self.nodes_index))
        )
        for i, a in enumerate(self.graph_input.nodes):
            for j, b in enumerate(self.graph_input.nodes):
                self.shortest_paths_arr[i, j] = self.shortest_paths_dict[a][b]
        self.nr_districts = nr_districts

        total_act = {}
        for act in activities:
            total_act[act] = 0
            for v in self.graph_input.nodes:
                total_act[act] += self.graph_input.nodes[v][act]

        self.totalAverageAct = {}
        for act in activities:
            self.totalAverageAct[act] = total_act[act] / nr_districts

    def remove_value(self, dictionary, value):
        for key, val in dictionary.items():
            if value in val:
                val.remove(value)

    def get_district_diameter_dict(self, district_nodes):
        solution_objective = 0
        incumbent_max = 0
        max_tuple = None
        for a in district_nodes:
            for b in district_nodes:
                incumbent_max = self.shortest_paths_dict[a][b]
                if incumbent_max > solution_objective:
                    solution_objective = incumbent_max
                    max_tuple = (a, b)
        return max_tuple, solution_objective

    def get_district_diameter_numpy(self, district_nodes, new_node=None):
        if district_nodes:
            nodes_idxs = [self.nodes_index[nd] for nd in district_nodes]
            if new_node:
                dist_arr = self.shortest_paths_arr[self.nodes_index[new_node]][nodes_idxs]
                max_to_idx = np.argmax(dist_arr)
                max_tuple = (new_node, district_nodes[max_to_idx])
            else:
                dist_arr = self.shortest_paths_arr[nodes_idxs][:, nodes_idxs]
                max_tuple_idx = np.unravel_index(np.argmax(dist_arr, axis=None), dist_arr.shape)
                max_tuple = (district_nodes[max_tuple_idx[0]], district_nodes[max_tuple_idx[1]])
            diameter = self.shortest_paths_dict[max_tuple[0]][max_tuple[1]]
            return {"diameter_nodes": max_tuple, "diameter": diameter}
        else:
            return {"diameter_nodes": None, "diameter": 0}

    def get_distance_to_district_numpy(self, from_node, district_nodes):
        if district_nodes:
            nodes_idxs = [self.nodes_index[nd] for nd in district_nodes]
            dist_arr = self.shortest_paths_arr[self.nodes_index[from_node]][nodes_idxs]
            min_to_idx = np.argmin(dist_arr)
            min_tuple = (from_node, district_nodes[min_to_idx])
            min_distance = self.shortest_paths_dict[min_tuple[0]][min_tuple[1]]
            return {"distance_to": min_tuple[1], "distance": min_distance}
        else:
            return {"distance_to": None, "distance": 0}

    def select_centroids(self):
        current_node = random.choice(self.nodes)
        centroids = [current_node]
        distances_to_centroids = self.shortest_paths_dict[current_node].copy()
        while len(centroids) < self.nr_districts:
            furthest_node = max(distances_to_centroids, key=distances_to_centroids.get)
            for node in self.nodes:
                if node not in centroids:
                    distance = self.shortest_paths_dict[node][furthest_node]
                    distances_to_centroids[node] = min(distances_to_centroids[node], distance)
            centroids.append(furthest_node)
        return centroids

    def get_furthest_nodes(self):
        sp_length = {}
        diameter = None
        furthest_node_list = []
        for node in self.nodes:
            sp_length[node] = nx.single_source_dijkstra_path_length(
                self.graph_input, node, weight="distance"
            )
            longest_path = max(sp_length[node].values())
            if diameter is None:
                diameter = longest_path
            if longest_path >= diameter:
                diameter = longest_path
                node_longest_paths = [
                    (node, other_node)
                    for other_node in sp_length[node].keys()
                    if sp_length[node][other_node] == longest_path
                ]
                if longest_path > diameter:
                    furthest_node_list = node_longest_paths
                else:
                    furthest_node_list = furthest_node_list + node_longest_paths
        return {
            "diameter": diameter,
            "furthest_node_list": furthest_node_list,
            "node_shortest_path_dicts": sp_length,
        }

    def get_district_center(self, district_nodes):
        if district_nodes:
            nodes_idxs = [self.nodes_index[nd] for nd in district_nodes]
            dist_arr = self.shortest_paths_arr[nodes_idxs][:, nodes_idxs]
            return self.nodes[nodes_idxs[np.argmin(np.max(dist_arr, axis=0))]]
        return None

    def get_district_infeasibility(self, district_nodes, activity_totals=None):
        if not activity_totals:
            activity_totals = {
                act: sum(self.graph_input.nodes[node][act] for node in district_nodes)
                for act in activities
            }
        infeasibility = 0
        for act in activities:
            infeasibility += (1 / self.totalAverageAct[act]) * max(
                activity_totals[act] - (1 + self.delta) * self.totalAverageAct[act],
                (1 - self.delta) * self.totalAverageAct[act] - activity_totals[act],
                0,
            )
        return infeasibility, activity_totals

    def constructDistricts(self, percentage_nodes=0.5, q=3, L=2):
        unassigned = set(self.nodes)
        district_act = {}
        district = {}
        for k, center_node in enumerate(self.centers_depots):
            district_act[k] = {}
            district[k] = [center_node]
            for act in activities:
                district_act[k][act] = self.graph_input.nodes[center_node][act]
            unassigned.remove(center_node)

        iteration = 0
        while (1 - percentage_nodes) * len(self.nodes) <= len(unassigned):
            for k, center_node in enumerate(self.centers_depots):
                unasigned_idx = [self.nodes_index[nd] for nd in unassigned]
                distances_to_unassigned = self.shortest_paths_arr[self.nodes_index[center_node]][unasigned_idx]
                q_nodes_idx = np.argpartition(distances_to_unassigned, q)[:q]
                chosen_nodes = np.take(list(unassigned), q_nodes_idx)
                for nd in chosen_nodes:
                    district[k].append(nd)
                    for act in activities:
                        district_act[k][act] += self.graph_input.nodes[nd][act]
                    unassigned.remove(nd)
            iteration += 1
            if iteration % L == 0:
                for k, center_node in enumerate(self.centers_depots):
                    district_nodes_idx = [self.nodes_index[nd] for nd in district[k]]
                    district_distances = self.shortest_paths_arr[district_nodes_idx][:, district_nodes_idx]
                    max_distances = np.max(district_distances, axis=0)
                    self.centers_depots[k] = self.nodes[district_nodes_idx[np.argmin(max_distances)]]

        max_dispersion = max(
            self.get_district_diameter_numpy(district[k])["diameter"] for k in district
        )
        dispersion = {}
        for k in district:
            dispersion[k] = {}
            for v in unassigned:
                new_diameter = self.get_district_diameter_numpy(district[k], v)["diameter"]
                dispersion[k][v] = max(max_dispersion, new_diameter) / self.graph_diameter

        infeasible = {}
        for k in district:
            infeasible[k] = {}
            for v in unassigned:
                infeasible[k][v] = 0
                for act in activities:
                    infeasible[k][v] += (1 / self.totalAverageAct[act]) * max(
                        district_act[k][act] + self.graph_input.nodes[v][act]
                        - (1 + self.delta) * self.totalAverageAct[act],
                        0,
                    )

        open_district = [True] * len(self.centers_depots)
        while unassigned and sum(open_district) > 0:
            for k, center_node in enumerate(self.centers_depots):
                if open_district[k]:
                    unassigned_neighbours = set().union(
                        *[unassigned.intersection(self.graph_input.adj[nd]) for nd in district[k]]
                    )
                    if unassigned_neighbours:
                        phi = {}
                        for v in unassigned_neighbours:
                            phi[v] = self.llambda * dispersion[k][v] + (1 - self.llambda) * infeasible[k][v]
                        phi_min = min(phi.values())
                        phi_max = max(phi.values())
                        RCL = [v for v in unassigned_neighbours
                               if phi[v] <= phi_min + self.rcl_parameter * (phi_max - phi_min)]
                        chosenRCL = random.choice(RCL)
                        district[k].append(chosenRCL)
                        for act in activities:
                            district_act[k][act] += self.graph_input.nodes[chosenRCL][act]
                        unassigned.remove(chosenRCL)
                        unassigned_neighbours.remove(chosenRCL)
                        activitiesCheck = 0
                        for act in activities:
                            activitiesCheck += (
                                sum(self.graph_input.nodes[nd][act] for nd in district[k])
                                >= (1 + self.delta) * self.totalAverageAct[act]
                            )
                        if activitiesCheck:
                            open_district[k] = False
                    else:
                        open_district[k] = False

        for node in unassigned:
            chosenDistrict = min(
                ((k, self.get_distance_to_district_numpy(node, dist_nodes)["distance"])
                 for k, dist_nodes in district.items()),
                key=lambda x: x[1],
            )[0]
            district[chosenDistrict].append(node)
            for act in activities:
                district_act[chosenDistrict][act] += self.graph_input.nodes[node][act]

        max_diameter = max(
            self.get_district_diameter_numpy(district[k])["diameter"] for k in district
        )
        solution_inf = sum(
            self.get_district_infeasibility(district[k], district_act[k])[0] for k in district
        )
        return max_diameter, solution_inf, district, district_act

    def evaluate_move(self, current_districts, node_to_move, from_district, to_district,
                      current_dist_diameters, current_dist_activities, current_infeasibilities,
                      lmbd1, lmbd2):
        other_max_diameter = max(
            current_dist_diameters[distr]["diameter"]
            for distr in current_dist_diameters
            if distr not in [from_district, to_district]
        )
        new_from_diameter = current_dist_diameters[from_district]
        if node_to_move in current_dist_diameters[from_district]["diameter_nodes"]:
            new_from_diameter = self.get_district_diameter_numpy(
                list(set(current_districts[from_district]) - {node_to_move})
            )
        new_to_diameter = current_dist_diameters[to_district]
        tmp_to_diameter = self.get_district_diameter_numpy(current_districts[to_district], node_to_move)
        if tmp_to_diameter["diameter"] > new_to_diameter["diameter"]:
            new_to_diameter = tmp_to_diameter
        new_diameter = max(other_max_diameter, new_from_diameter["diameter"], new_to_diameter["diameter"])

        new_infeasibility = sum(
            current_infeasibilities[k]
            for k in current_dist_activities
            if k not in [from_district, to_district]
        )
        new_activity_totals = {from_district: {}, to_district: {}}
        for act in activities:
            new_activity_totals[from_district][act] = (
                current_dist_activities[from_district][act] - self.graph_input.nodes[node_to_move][act]
            )
            new_activity_totals[to_district][act] = (
                current_dist_activities[to_district][act] + self.graph_input.nodes[node_to_move][act]
            )
        new_infeasibility += sum(
            self.get_district_infeasibility(current_districts[k], new_activity_totals[k])[0]
            for k in [from_district, to_district]
        )
        new_merit = lmbd1 * new_diameter / self.graph_diameter + lmbd2 * new_infeasibility
        return (new_merit, new_diameter, new_infeasibility,
                {from_district: new_from_diameter, to_district: new_to_diameter},
                new_activity_totals)

    def localSearch(self, initial_objective, initial_infeasibility, initial_solution,
                    limit_evals=1000, use_adjacency=False):
        current_best_solution = cp.deepcopy(initial_solution)
        current_best_merit = (
            self.llambda * (initial_objective / self.graph_diameter)
            + (1 - self.llambda) * initial_infeasibility
        )
        current_best_objective = initial_objective
        current_best_infeasibility = initial_infeasibility

        node_district_matching = {}
        for k in current_best_solution:
            for nd in current_best_solution[k]:
                node_district_matching[nd] = k

        current_diameters = {i: self.get_district_diameter_numpy(initial_solution[i]) for i in initial_solution}

        current_dist_activities = {}
        for k in current_best_solution:
            current_dist_activities[k] = {
                act: sum(self.graph_input.nodes[nd][act] for nd in current_best_solution[k])
                for act in activities
            }

        current_infeasibilities = {
            k: self.get_district_infeasibility(current_best_solution[k], current_dist_activities[k])[0]
            for k in current_best_solution
        }

        nmoves = 0
        local_optima = False
        k = 0
        kend = max(current_best_solution.keys())

        if use_adjacency:
            moves = {
                i: {(nd, node_district_matching[d])
                    for nd in current_best_solution[i]
                    for d in self.graph_input.adj[nd]
                    if d not in current_best_solution[i]}
                for i in current_best_solution
            }
        else:
            moves = {
                i: {(nd, d) for nd in current_best_solution[i] for d in current_best_solution if d != i}
                for i in current_best_solution
            }

        while nmoves <= limit_evals and not local_optima:
            improvement = False
            while moves[k] and not improvement:
                move_to_test = moves[k].pop()
                (new_merit, new_max_diameter, new_infeasibility,
                 new_diameters, new_activity_totals) = self.evaluate_move(
                    current_best_solution, move_to_test[0], k, move_to_test[1],
                    current_diameters, current_dist_activities, current_infeasibilities,
                    self.llambda, 1 - self.llambda,
                )
                if new_merit < current_best_merit:
                    best_move = move_to_test
                    current_best_merit = new_merit
                    current_best_objective = new_max_diameter
                    current_best_infeasibility = new_infeasibility
                    best_new_diameters = new_diameters
                    best_new_activity_totals = new_activity_totals
                    improvement = True

                    new_district = best_move[1]
                    current_best_solution[new_district].append(best_move[0])
                    current_best_solution[k].remove(best_move[0])
                    node_district_matching[best_move[0]] = new_district

                    current_diameters[k] = best_new_diameters[k]
                    current_diameters[new_district] = best_new_diameters[new_district]
                    current_dist_activities[k] = best_new_activity_totals[k]
                    current_dist_activities[new_district] = best_new_activity_totals[new_district]

                    moves[k] -= {mv for mv in moves[k] if mv[0] == best_move[0]}
                    if use_adjacency:
                        moves[k].update(
                            {(d, node_district_matching[best_move[0]])
                             for d in self.graph_input.adj[best_move[0]]
                             if d in current_best_solution[k]}
                        )
                        moves[new_district] -= {
                            mv for mv in moves[new_district]
                            if mv[1] in self.graph_input.adj[best_move[0]]
                        }
                        moves[new_district].update(
                            {(best_move[0], node_district_matching[d])
                             for d in self.graph_input.adj[best_move[0]]
                             if d not in current_best_solution[new_district]}
                        )
                    else:
                        moves[new_district].update(
                            {(best_move[0], d) for d in current_best_solution if d != new_district}
                        )

                    current_infeasibilities[k] = self.get_district_infeasibility(
                        current_best_solution[k], current_dist_activities[k])[0]
                    current_infeasibilities[new_district] = self.get_district_infeasibility(
                        current_best_solution[new_district], current_dist_activities[new_district])[0]

                    kend = k
                    k = (k + 1) % len(current_dist_activities)
                    nmoves += 1

            if not improvement:
                k = (k + 1) % len(current_dist_activities)
            if k == kend and not moves[k]:
                local_optima = True

        return current_best_objective, current_best_infeasibility, current_best_solution
