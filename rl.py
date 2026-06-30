import itertools
import networkx as nx
import copy as cp
import numpy as np
import random
import bisect
from collections import deque
from scipy.optimize import linear_sum_assignment

import time

import logging

logging.basicConfig(level=logging.WARNING)


activities = ["workload", "n_customers", "demand"]


class TerritoryDesignProblem:
    """
    A class representing the territory design problem.

    Parameters:
    -----------
    graph_input : networkx.Graph
        A graph representing the road network.
    delta : float
        A value used for calculating infeasibility.
    llambda : float
        A weight parameter for the objective function.
    rcl_parameter : float
        A parameter used for generating restricted candidate lists.
    nr_districts : int
        The number of districts to create

    Attributes:
    -----------
    graph_input : networkx.Graph
        A graph representing the road network.
    delta : float
        A value used for calculating infeasibility.
    llambda : float
        A weight parameter for the objective function.
    rcl_parameter : float
        A parameter used for generating restricted candidate lists.
    nodes : list
        A list of nodes in the graph.
    centers_depots : list
        A list of centers/depots.
    adjacent_nodes : dict
        A dictionary of adjacent nodes for each node in the graph.
    originalAdjacentNodes : dict
        A deepcopy of adjacent_nodes.
    shortest_paths_dict : dict
        A dictionary of shortest paths between each pair of nodes in the graph.
    graph_diameter : int
        The diameter of the graph.
    unassigned : list
        A list of nodes that have not been assigned to any district.
    average_workload : float
        The average workload of centers/depots.
    average_customers : float
        The average number of customers of centers/depots.
    average_demand : float
        The average demand of centers/depots.

    Methods:
    --------
    constructDistricts(self):
        A method to implement a greedy algorithm for constructing districts.
    update_merit_function(self,input_solution,k,l,bu):
        A method for computing the merit function for moving a node bu from district k to district l of the
        input_solution.
    nodeBestImprovement(self,initial_solution,initial_objective,initial_infeasibility):
        A method for implementing a node best improvement local search algorithm using the intial solution,
        its objective, and infeasibility
    depotBestImprovement(self,initial_solution,initial_objective,initial_infeasibility):
        A method for implementing a depot best improvement local search algorithm using the intial solution,
        its objective, and infeasibility


    """

    def __init__(self, graph_input, delta, llambda, rcl_parameter, nr_districts=10):

        self.graph_input = graph_input
        self.delta = delta
        self.llambda = llambda
        self.rcl_parameter = rcl_parameter
        self.nodes = list(graph_input.nodes())
        getFurthestNodesResponse = self.get_furthest_nodes()  # self.graph_input)
        self.shortest_paths_dict = getFurthestNodesResponse["node_shortest_path_dicts"]
        self.graph_diameter = getFurthestNodesResponse["diameter"]
        self.nodes_index = {v: i for i, v in enumerate(self.graph_input.nodes)}
        # print(self.nodes_index)
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
        # Helper utility to drop a node from all adjacency entries that still reference it.
        for key, val in dictionary.items():
            if value in val:
                val.remove(value)

    def get_district_diameter_dict(self, district_nodes):
        # Exhaustively compare each ordered pair when districts are small enough for a brute-force pass.
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
        # Speed up repeated diameter queries using the APSP matrix and optional candidate insertion.
        if district_nodes:
            nodes_idxs = [self.nodes_index[nd] for nd in district_nodes]
            if new_node:
                dist_arr = self.shortest_paths_arr[self.nodes_index[new_node]][
                    nodes_idxs
                ]
                max_to_idx = np.argmax(dist_arr)
                max_tuple = (new_node, district_nodes[max_to_idx])
            else:
                dist_arr = self.shortest_paths_arr[nodes_idxs][:, nodes_idxs]
                max_tuple_idx = np.unravel_index(
                    np.argmax(dist_arr, axis=None), dist_arr.shape
                )
                max_tuple = (
                    district_nodes[max_tuple_idx[0]],
                    district_nodes[max_tuple_idx[1]],
                )

            diameter = self.shortest_paths_dict[max_tuple[0]][max_tuple[1]]

            return {"diameter_nodes": max_tuple, "diameter": diameter}
        else:
            return {"diameter_nodes": None, "diameter": 0}

    def get_distance_to_district_numpy(self, from_node, district_nodes):
        # Obtain the nearest neighbor in a district via a single vectorized row lookup.
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
        """
        Returns a list of 'k' centroids by selecting the farthest nodes from 'nodes' iteratively.

        Returns:
        list: A list containing 'num_centroids' centroids.
        """
        # Select a random node to start from
        current_node = random.choice(self.nodes)
        centroids = [current_node]

        # Calculate the initial distances from the current_node to all other nodes
        distances_to_centroids = self.shortest_paths_dict[current_node].copy()

        # Find furthest nodes until desired number of centroids is reached
        while len(centroids) < self.nr_districts:
            # Find the node furthest away from all other centroids
            furthest_node = max(distances_to_centroids, key=distances_to_centroids.get)

            # Update distances_to_centroids with the new furthest_node
            for node in self.nodes:
                if node not in centroids:
                    # distance = nx.shortest_path_length(G, node, furthest_node, weight='distance')
                    distance = self.shortest_paths_dict[node][furthest_node]
                    distances_to_centroids[node] = min(
                        distances_to_centroids[node], distance
                    )

            # Add the furthest_node to the list of centroids
            centroids.append(furthest_node)

        return centroids

    def get_furthest_nodes(self):
        """
        Returns the diameter, all pairs of nodes with shortest path length equal to the diameter,
        and the dict of all-node shortest paths of a graph 'G'.

        Returns:
        dict: A dictionary containing 'diameter', 'furthest_node_list', and 'node_shortest_path_dicts'.
        """

        sp_length = {}  # dict containing shortest path distances for each pair of nodes
        diameter = (
            None  # will contain the graphs diameter (length of longest shortest path)
        )
        furthest_node_list = (
            []
        )  # will contain list of tuple of nodes with shortest path equal to diameter

        for node in self.nodes:
            # Get the shortest path from node to all other nodes
            sp_length[node] = nx.single_source_dijkstra_path_length(
                self.graph_input, node, weight="distance"
            )
            longest_path = max(
                sp_length[node].values()
            )  # get length of furthest node from node

            # Update diameter when necessary (on first iteration and when we find a longer one)
            if diameter == None:
                diameter = longest_path  # set the first diameter

            # update the list of tuples of furthest nodes if we have a best diameter
            if longest_path >= diameter:
                diameter = longest_path

                # a list of tuples containing
                # the current node and the nodes furthest from it
                node_longest_paths = [
                    (node, other_node)
                    for other_node in sp_length[node].keys()
                    if sp_length[node][other_node] == longest_path
                ]
                if longest_path > diameter:
                    # This is better than the previous diameter
                    # so replace the list of tuples of diameter nodes with this nodes
                    # tuple of furthest nodes
                    furthest_node_list = node_longest_paths
                else:  # this is equal to the current diameter
                    # add this nodes tuple of furthest nodes to the current list
                    furthest_node_list = furthest_node_list + node_longest_paths

        # return the diameter,
        # all pairs of nodes with shortest path length equal to the diameter
        # the dict of all-node shortest paths
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
        else:
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
        # Greedily grow districts from current centers while tracking activity totals for balance.
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
                # get q nearest unasigned neighbours
                unasigned_idx = [self.nodes_index[nd] for nd in unassigned]
                distances_to_unassigned = self.shortest_paths_arr[
                    self.nodes_index[center_node]
                ][unasigned_idx]
                q_nodes_idx = np.argpartition(distances_to_unassigned, q)[:q]
                chosen_nodes = np.take(list(unassigned), q_nodes_idx)
                for nd in chosen_nodes:
                    district[k].append(nd)
                    for act in activities:
                        district_act[k][act] += self.graph_input.nodes[nd][act]
                    unassigned.remove(nd)

            iteration += 1
            if iteration % L == 0:  # update centers
                for k, center_node in enumerate(self.centers_depots):
                    district_nodes_idx = [self.nodes_index[nd] for nd in district[k]]
                    district_distances = self.shortest_paths_arr[district_nodes_idx][
                        :, district_nodes_idx
                    ]
                    max_distances = np.max(district_distances, axis=0)
                    self.centers_depots[k] = self.nodes[
                        district_nodes_idx[np.argmin(max_distances)]
                    ]

        max_dispersion = max(
            self.get_district_diameter_numpy(district[k])["diameter"] for k in district
        )
        dispersion = {}
        for k in district:
            dispersion[k] = {}
            for v in unassigned:
                new_diameter = self.get_district_diameter_numpy(district[k], v)[
                    "diameter"
                ]
                dispersion[k][v] = (
                    max(max_dispersion, new_diameter) / self.graph_diameter
                )

        infeasible = {}
        for k in district:
            infeasible[k] = {}
            for v in unassigned:
                infeasible[k][v] = 0
                for act in activities:
                    infeasible[k][v] += (1 / self.totalAverageAct[act]) * max(
                        district_act[k][act]
                        + self.graph_input.nodes[v][act]
                        - (1 + self.delta) * self.totalAverageAct[act],
                        0,
                    )

        open_district = [True] * len(self.centers_depots)

        while unassigned and sum(open_district) > 0:
            for k, center_node in enumerate(self.centers_depots):
                if open_district[k]:
                    # Candidate queue is limited to frontier nodes to maintain spatial compactness.
                    unassigned_neighbours = set().union(
                        *[
                            unassigned.intersection(self.graph_input.adj[nd])
                            for nd in district[k]
                        ]
                    )

                    if unassigned_neighbours:
                        phi = {}
                        for v in unassigned_neighbours:
                            phi[v] = (
                                self.llambda * dispersion[k][v]
                                + (1 - self.llambda) * infeasible[k][v]
                            )
                        phi_min = min(phi.values())
                        phi_max = max(phi.values())
                        RCL = [
                            v
                            for v in unassigned_neighbours
                            if phi[v]
                            <= phi_min + self.rcl_parameter * (phi_max - phi_min)
                        ]
                        chosenRCL = random.choice(RCL)
                        district[k].append(chosenRCL)
                        for act in activities:
                            district_act[k][act] += self.graph_input.nodes[chosenRCL][
                                act
                            ]
                        unassigned.remove(chosenRCL)
                        unassigned_neighbours.remove(chosenRCL)
                        activitiesCheck = 0
                        for act in activities:
                            activitiesCheck += (
                                sum(
                                    self.graph_input.nodes[nd][act]
                                    for nd in district[k]
                                )
                                >= (1 + self.delta) * self.totalAverageAct[act]
                            )
                        if activitiesCheck:
                            open_district[k] = False
                    else:
                        open_district[k] = False

        for node in unassigned:
            # If no frontier remained, attach free nodes to their closest feasible district.
            chosenDistrict = min(
                (
                    (
                        k,
                        self.get_distance_to_district_numpy(node, dist_nodes)[
                            "distance"
                        ],
                    )
                    for k, dist_nodes in district.items()
                ),
                key=lambda x: x[1],
            )[0]
            district[chosenDistrict].append(node)
            for act in activities:
                district_act[chosenDistrict][act] += self.graph_input.nodes[node][act]

        max_diameter = max(
            self.get_district_diameter_numpy(district[k])["diameter"] for k in district
        )

        solution_inf = sum(
            self.get_district_infeasibility(district[k], district_act[k])[0]
            for k in district
        )

        # for act in activities:
        #     for k in district:
        #         solution_inf += (1/self.average_act[act])*max(district_act[act][k] - (1+self.delta)*self.average_act[act], \
        #                                                      (1-self.delta)*self.average_act[act] - district_act[act][k], \
        #                                                      0)

        return max_diameter, solution_inf, district, district_act

    def evaluate_move(
        self,
        current_districts,
        node_to_move,
        from_district,
        to_district,
        current_dist_diameters,
        current_dist_activities,
        current_infeasibilities,
        lmbd1,
        lmbd2,
    ):
        """
        function calculates the merit function of a given district allocation by updating a
        given solution input_solution with a new basic unit bu that moves from district l to
        district k. It returns the updated merit function, objective value, and infeasibility
        of the updated solution.

        Parameters:
        -----------
        input_solution: dict
            A dictionary representing the initial district allocation, where the keys are the
            centers/depots and the values are the list of basic units that belong to that center/depot.
        to_district: int
            The index of the center/depot that will receive the new 'node_to_move'.
        from_district: int
            The index of the center/depot that currently owns the 'node_to_move'.
        node_to_move: int
            The index of the node that will be moved from "from_district" to 'to_district'.

        current_dist_diameters: dict of floats
            diameters per distict

        current_infeasibilities: dict of floats
            infeasibilities per district

        Returns:
        --------
        new_merit: float
            The updated merit function of the new solution with the moved 'node_to_move'.
        new_diameter:
            int The objective value of the updated solution.
        new_infeasibility:
            float The infeasibility of the updated solution.
        """

        # Base diameter equals the best of the districts unaffected by the potential move.
        other_max_diameter = max(
            current_dist_diameters[distr]["diameter"]
            for distr in current_dist_diameters
            if distr not in [from_district, to_district]
        )

        new_from_diameter = current_dist_diameters[from_district]
        if node_to_move in current_dist_diameters[from_district]["diameter_nodes"]:
            # if the node_to_move is on the edge of the diameter
            new_from_diameter = self.get_district_diameter_numpy(
                list(set(current_districts[from_district]) - {node_to_move})
            )

        # measure the max distance from the "node_to_move" to "to_district" and compare with the curent district diameter
        new_to_diameter = current_dist_diameters[to_district]
        tmp_to_diameter = self.get_district_diameter_numpy(
            current_districts[to_district], node_to_move
        )
        if tmp_to_diameter["diameter"] > new_to_diameter["diameter"]:
            new_to_diameter = tmp_to_diameter

        new_diameter = max(
            other_max_diameter,
            new_from_diameter["diameter"],
            new_to_diameter["diameter"],
        )

        # Preserve infeasibility of untouched districts and only recompute the two that change.
        new_infeasibility = sum(
            current_infeasibilities[k]
            for k in current_dist_activities
            if k not in [from_district, to_district]
        )

        new_activity_totals = {}
        new_activity_totals[from_district] = {}
        new_activity_totals[to_district] = {}
        for act in activities:
            new_activity_totals[from_district][act] = (
                current_dist_activities[from_district][act]
                - self.graph_input.nodes[node_to_move][act]
            )
            new_activity_totals[to_district][act] = (
                current_dist_activities[to_district][act]
                + self.graph_input.nodes[node_to_move][act]
            )

        new_infeasibility += sum(
            self.get_district_infeasibility(
                current_districts[k], new_activity_totals[k]
            )[0]
            for k in [from_district, to_district]
        )

        new_merit = (
            lmbd1 * new_diameter / self.graph_diameter + lmbd2 * new_infeasibility
        )

        return (
            new_merit,
            new_diameter,
            new_infeasibility,
            {from_district: new_from_diameter, to_district: new_to_diameter},
            new_activity_totals,
        )

    def moveNode(
        self,
        districts,
        node_to_move,
        dist_from,
        dist_to,
        node_district_matching,
        adjacentDistricts,
        border_edges,
    ):
        node_district_matching[node_to_move] = dist_to
        districts[dist_from].remove(node_to_move)
        districts[dist_to].append(node_to_move)

        # Update the edge frontier for both districts because the moved node swaps sides.
        border_edges[dist_from] -= {
            edg for edg in border_edges[dist_from] if border_edges[0] == node_to_move
        }
        border_edges[dist_from].update(
            {
                (nd, node_to_move)
                for nd in self.graph_input.adj[node_to_move]
                if nd in districts[dist_from]
            }
        )
        border_edges[dist_to] -= {
            edg for edg in border_edges[dist_to] if border_edges[1] == node_to_move
        }
        border_edges[dist_to].update(
            {
                (node_to_move, nd)
                for nd in self.graph_input.adj[node_to_move]
                if not nd in districts[dist_to]
            }
        )

        self.remove_value(adjacentDistricts[dist_from], node_to_move)

        keys_to_remove = []

        for depot in adjacentDistricts[dist_from]:
            if len(adjacentDistricts[dist_from][depot]) == 0:
                keys_to_remove.append(depot)
        for key in keys_to_remove:
            adjacentDistricts[dist_from].pop(key)

        for adjacent_node in self.graph_input.adj[
            node_to_move
        ]:  # self.originalAdjacentNodes[node_to_move]:
            # Track which boundary nodes now expose districts to each other after the move.
            if node_district_matching[adjacent_node] != dist_to:
                if node_district_matching[adjacent_node] in adjacentDistricts[dist_to]:
                    adjacentDistricts[dist_to][
                        node_district_matching[adjacent_node]
                    ].add(node_to_move)
                else:
                    adjacentDistricts[dist_to][
                        node_district_matching[adjacent_node]
                    ] = set()
                    adjacentDistricts[dist_to][
                        node_district_matching[adjacent_node]
                    ].add(node_to_move)

            if node_district_matching[adjacent_node] == dist_from:
                if dist_to in adjacentDistricts[dist_from]:
                    adjacentDistricts[dist_from][dist_to].add(adjacent_node)
                else:
                    adjacentDistricts[dist_from][dist_to] = set()
                    adjacentDistricts[dist_from][dist_to].add(adjacent_node)

        for connected_district in list(adjacentDistricts[dist_to]):
            for connecting_node in list(adjacentDistricts[dist_to][connected_district]):
                if all(
                    node_district_matching[n] == dist_to
                    for n in self.graph_input.adj[connecting_node]
                ):
                    adjacentDistricts[dist_to][connected_district].remove(
                        connecting_node
                    )
                    if len(adjacentDistricts[dist_to][connected_district]) == 0:
                        adjacentDistricts[dist_to].pop(connected_district)

        for connected_district in list(adjacentDistricts[dist_from]):
            for connecting_node in list(
                adjacentDistricts[dist_from][connected_district]
            ):
                if all(
                    node_district_matching[n] == dist_from
                    for n in self.graph_input.adj[connecting_node]
                ):
                    adjacentDistricts[dist_from][connected_district].remove(
                        connecting_node
                    )
                    if len(adjacentDistricts[dist_from][connected_district]) == 0:
                        adjacentDistricts[dist_from].pop(connected_district)

        return districts, node_district_matching, adjacentDistricts

    def localSearch(
        self,
        initial_objective,
        initial_infeasibility,
        initial_solution,
        limit_evals=1000,
        use_adjacency=False,
    ):
        """
        The localSearch function searches for the best move of a node adjacent to district k from
        its current district to district k, and updates the current solution and the corresponding
        objective and infeasibility values. The search stops when it reaches the maximum number of moves,
        or when no improvement can be made.

        Parameters:
        -----------
        initial_solution: dict
            A dictionary representing the initial district allocation, where the keys are the centers/depots
            and the values are the list of basic units that belong to that center/depot.
        initial_objective: int
            The objective value of the initial solution.
        initial_infeasibility: float
            The infeasibility of the initial solution.

        Returns:
        --------
        current_best_objective: int
            The objective value of the best solution found.
        current_best_infeasibility: float
            The infeasibility of the best solution found.
        current_best_solution: dict
            A dictionary representing the best district allocation found, where the keys are the centers/depots
            and the values are the list of basic units that belong to that center/depot.

        """

        current_best_solution = cp.deepcopy(initial_solution)
        current_best_merit = (
            self.llambda * (initial_objective / self.graph_diameter)
            + (1 - self.llambda) * initial_infeasibility
        )
        current_best_objective = initial_objective
        current_best_infeasibility = initial_infeasibility

        node_district_matching = {}
        # Maintain a fast lookup for the district owning each node to avoid repeated scans.
        for k in current_best_solution:
            for nd in current_best_solution[k]:
                node_district_matching[nd] = k

        current_diameters = {
            i: self.get_district_diameter_numpy(initial_solution[i])
            for i in initial_solution
        }

        current_dist_activities = {}
        for k in current_best_solution:
            current_dist_activities[k] = {}
            for act in activities:
                current_dist_activities[k][act] = sum(
                    self.graph_input.nodes[nd][act] for nd in current_best_solution[k]
                )

        current_infeasibilities = {}
        for k in current_best_solution:
            current_infeasibilities[k] = self.get_district_infeasibility(
                current_best_solution[k], current_dist_activities[k]
            )[0]

        nmoves = 0
        local_optima = False
        k = 0
        kend = max(current_best_solution.keys())

        if use_adjacency:
            # Neighborhood restricted to physically adjacent swaps to accelerate convergence.
            moves = {
                i: {
                    (nd, node_district_matching[d])
                    for nd in current_best_solution[i]
                    for d in self.graph_input.adj[nd]
                    if not d in current_best_solution[i]
                }
                for i in current_best_solution
            }
        else:
            moves = {
                i: {
                    (nd, d)
                    for nd in current_best_solution[i]
                    for d in current_best_solution
                    if d != i
                }
                for i in current_best_solution
            }

        while nmoves <= limit_evals and not local_optima:
            improvement = False
            # collect adjacent nodes do district k

            while moves[k] and not improvement:
                # Choose valid move from N(Xk ); N(Xk ) ← N(Xk ) \ {(i, j)}

                move_to_test = moves[k].pop()

                # Compute the effect of moving the node between districts without mutating the current solution.
                (
                    new_merit,
                    new_max_diameter,
                    new_infeasibility,
                    new_diameters,
                    new_activity_totals,
                ) = self.evaluate_move(
                    current_best_solution,
                    move_to_test[0],
                    k,
                    move_to_test[1],
                    current_diameters,
                    current_dist_activities,
                    current_infeasibilities,
                    self.llambda,
                    1 - self.llambda,
                )

                if new_merit < current_best_merit:
                    best_move = move_to_test
                    current_best_merit = new_merit
                    current_best_objective = new_max_diameter
                    current_best_infeasibility = new_infeasibility
                    best_new_diameters = new_diameters
                    best_new_activity_totals = new_activity_totals
                    improvement = True

                    # perform move

                    new_district = best_move[1]
                    current_best_solution[new_district].append(best_move[0])
                    current_best_solution[k].remove(best_move[0])
                    node_district_matching[best_move[0]] = new_district

                    current_diameters[k] = best_new_diameters[k]
                    current_diameters[new_district] = best_new_diameters[new_district]

                    current_dist_activities[k] = best_new_activity_totals[k]
                    current_dist_activities[new_district] = best_new_activity_totals[
                        new_district
                    ]

                    moves[k] -= {mv for mv in moves[k] if mv[0] == best_move[0]}
                    if use_adjacency:
                        moves[k].update(
                            {
                                (d, node_district_matching[best_move[0]])
                                for d in self.graph_input.adj[best_move[0]]
                                if d in current_best_solution[k]
                            }
                        )
                        moves[new_district] -= {
                            mv
                            for mv in moves[new_district]
                            if mv[1] in self.graph_input.adj[best_move[0]]
                        }
                        moves[new_district].update(
                            {
                                (best_move[0], node_district_matching[d])
                                for d in self.graph_input.adj[best_move[0]]
                                if not d in current_best_solution[new_district]
                            }
                        )
                    else:
                        moves[new_district].update(
                            {
                                (best_move[0], d)
                                for d in current_best_solution
                                if d != new_district
                            }
                        )

                    current_infeasibilities[k] = self.get_district_infeasibility(
                        current_best_solution[k], current_dist_activities[k]
                    )[0]
                    current_infeasibilities[new_district] = (
                        self.get_district_infeasibility(
                            current_best_solution[new_district],
                            current_dist_activities[new_district],
                        )[0]
                    )

                    kend = k
                    k = (k + 1) % len(current_dist_activities)
                    nmoves += 1

            if not improvement:
                k = (k + 1) % len(current_dist_activities)

            if k == kend and not moves[k]:
                local_optima = True

        return current_best_objective, current_best_infeasibility, current_best_solution

    def nodeBestImprovement(
        self,
        initial_solution,
        initial_objective,
        initial_infeasibility,
        adjacentDistricts,
        border_edges,
        limit_evals=1000,
    ):
        """
        The nodeBestImprovement function searches for a better solution by moving one basic unit from
        its current district to another district, and updates the current solution and the corresponding
        objective and infeasibility values. The search stops when it reaches the maximum number of moves,
        or when no improvement can be made.

        Parameters:
        -----------
        initial_solution: dict
            A dictionary representing the initial district allocation, where the keys are the centers/depots
            and the values are the list of basic units that belong to that center/depot.
        initial_objective: int
            The objective value of the initial solution.
        initial_infeasibility: float
            The infeasibility of the initial solution.

        Returns:
        --------
        current_best_objective: int
            The objective value of the best solution found.
        current_best_infeasibility: float
            The infeasibility of the best solution found.
        current_best_solution: dict
            A dictionary representing the best district allocation found, where the keys are the centers/depots
            and the values are the list of basic units that belong to that center/depot.
        """

        current_best_solution = cp.deepcopy(initial_solution)
        current_best_merit = (
            initial_objective / self.graph_diameter
        ) + self.llambda * initial_infeasibility

        current_best_objective = initial_objective
        current_best_infeasibility = initial_infeasibility

        node_district_matching = {}
        for dist_idx in current_best_solution:
            for i in current_best_solution[dist_idx]:
                node_district_matching[i] = dist_idx

        moves = {}
        for dist_idx in current_best_solution:
            # Each district initially considers importing any node not currently assigned to it.
            moves[dist_idx] = [
                node
                for node in node_district_matching
                if node_district_matching[node] != dist_idx
            ]

        current_diameters = {
            i: self.get_district_diameter_numpy(initial_solution[i])
            for i in initial_solution
        }
        current_max_diameter = max(d["diameter"] for d in current_diameters.values())

        current_dist_activities = {}
        for dist in current_best_solution:
            current_dist_activities[dist] = {}
            current_dist_activities[dist]["n_customers"] = sum(
                self.graph_input.nodes[nd]["n_customers"]
                for nd in current_best_solution[dist]
            )
            current_dist_activities[dist]["workload"] = sum(
                self.graph_input.nodes[nd]["workload"]
                for nd in current_best_solution[dist]
            )
            current_dist_activities[dist]["demand"] = sum(
                self.graph_input.nodes[nd]["demand"]
                for nd in current_best_solution[dist]
            )

        current_infeasibilities = {}
        for k in current_best_solution:
            current_infeasibilities[k] = self.get_district_infeasibility(
                current_best_solution[k], current_dist_activities[k]
            )[0]

        nmoves = 0
        local_optima = False

        while nmoves < limit_evals and not local_optima:

            improvement = False
            for k in initial_solution:
                districtImprovement = False

                for node_to_move in moves[k]:

                    # Evaluate the move of every candidate node into district k.
                    (
                        new_merit,
                        new_max_diameter,
                        new_infeasibility,
                        new_diameters,
                        new_activity_totals,
                    ) = self.evaluate_move(
                        current_best_solution,
                        node_to_move,
                        node_district_matching[node_to_move],
                        k,
                        current_diameters,
                        current_dist_activities,
                        current_infeasibilities,
                        1,
                        self.llambda,
                    )

                    if new_merit < current_best_merit:
                        best_node_to_move = node_to_move
                        current_best_merit = new_merit
                        current_best_objective = new_max_diameter
                        current_best_infeasibility = new_infeasibility
                        best_new_diameters = new_diameters
                        best_new_activity_totals = new_activity_totals
                        improvement = True
                        districtImprovement = True

                if districtImprovement == True:

                    move_from = node_district_matching[best_node_to_move]

                    current_diameters[k] = cp.deepcopy(best_new_diameters[k])
                    current_diameters[move_from] = cp.deepcopy(
                        best_new_diameters[move_from]
                    )

                    current_dist_activities[k] = cp.deepcopy(
                        best_new_activity_totals[k]
                    )
                    current_dist_activities[move_from] = cp.deepcopy(
                        best_new_activity_totals[move_from]
                    )

                    moves[k].remove(best_node_to_move)
                    moves[move_from].append(best_node_to_move)

                    current_best_solution, node_district_matching, adjacentDistricts = (
                        self.moveNode(
                            current_best_solution,
                            best_node_to_move,
                            move_from,
                            k,
                            node_district_matching,
                            adjacentDistricts,
                            border_edges,
                        )
                    )

                    current_infeasibilities[k] = self.get_district_infeasibility(
                        current_best_solution[k], current_dist_activities[k]
                    )[0]
                    current_infeasibilities[move_from] = (
                        self.get_district_infeasibility(
                            current_best_solution[move_from],
                            current_dist_activities[move_from],
                        )[0]
                    )

            if improvement == True:
                nmoves = nmoves + 1
                local_optima = False
            else:
                local_optima = True

        return (
            current_best_objective,
            current_best_infeasibility,
            current_best_solution,
            adjacentDistricts,
            border_edges,
        )

    def calculateInfeasibilityBreakdowns(self, district, delta=0.05):
        # Provide a per-activity infeasibility report to diagnose the worst balancing offsets.
        total = {}
        average = {}
        for act in activities:
            total[act] = 0
            for nd in self.graph_input.nodes:
                total[act] += self.graph_input.nodes[nd][act]
            average[act] = total[act] / len(district)

        district_total = {}
        for k in district:
            district_total[k] = {}
            for act in activities:
                district_total[k][act] = sum(
                    self.graph_input.nodes[nd][act] for nd in district[k]
                )

        infeasibility = {}
        for k in district:
            infeasibility[k] = {}
            for act in activities:
                infeasibility[k][act] = (1 / average[act]) * max(
                    district_total[k][act] - (1 + delta) * average[act],
                    (1 - delta) * average[act] - district_total[k][act],
                    0,
                )

        infeasibility_calculation = {
            "Customers Infeasibility": {
                k: infeasibility[k]["n_customers"] for k in district
            },
            "Demand Infeasibility": {k: infeasibility[k]["demand"] for k in district},
            "Workload Infeasibility": {
                k: infeasibility[k]["workload"] for k in district
            },
        }

        return infeasibility_calculation

    @staticmethod
    def evaluateSolution(allocation, graph_input, delta):

        centers_depots = list(allocation.keys())

        sp_all = {
            o: dists
            for (o, dists) in nx.shortest_path_length(graph_input, weight="distance")
        }

        max_diameters = {
            k: max(sp_all[o][d] for o in allocation[k] for d in allocation[k])
            for k in centers_depots
        }

        max_graph_diameter = max(max(dists.values()) for dists in sp_all.values())

        total_workload = sum(
            graph_input.nodes[v]["workload"] for v in graph_input.nodes
        )
        total_customers = sum(
            graph_input.nodes[v]["n_customers"] for v in graph_input.nodes
        )
        total_demand = sum(graph_input.nodes[v]["demand"] for v in graph_input.nodes)

        average_workload = total_workload / len(centers_depots)
        average_customers = total_customers / len(centers_depots)
        average_demand = total_demand / len(centers_depots)

        total_district_customers = {}
        total_district_workload = {}
        total_district_demand = {}

        for k in centers_depots:
            total_district_customers[k] = sum(
                graph_input.nodes[nd]["n_customers"] for nd in allocation[k]
            )
            total_district_demand[k] = sum(
                graph_input.nodes[nd]["demand"] for nd in allocation[k]
            )
            total_district_workload[k] = sum(
                graph_input.nodes[nd]["workload"] for nd in allocation[k]
            )

        infeasibility_customers = {}
        infeasibility_demand = {}
        infeasibility_workload = {}
        infeasibility_total = {}

        for k in centers_depots:
            infeasibility_customers[k] = (1 / average_customers) * max(
                total_district_customers[k] - (1 + delta) * average_customers,
                (1 - delta) * average_customers - total_district_customers[k],
                0,
            )
            infeasibility_demand[k] = (1 / average_demand) * max(
                total_district_demand[k] - (1 + delta) * average_demand,
                (1 - delta) * average_demand - total_district_demand[k],
                0,
            )
            infeasibility_workload[k] = (1 / average_workload) * max(
                total_district_workload[k] - (1 + delta) * average_workload,
                (1 - delta) * average_workload - total_district_workload[k],
                0,
            )
            infeasibility_total[k] = (
                infeasibility_customers[k]
                + infeasibility_demand[k]
                + infeasibility_workload[k]
            )

        return (
            max(max_diameters.values()),
            sum(infeasibility_customers.values())
            + sum(infeasibility_demand.values())
            + sum(infeasibility_workload.values()),
            {
                "graph diameter": max_graph_diameter,
                "district Diameters": max_diameters,
                "activity totals": {
                    "n_customers": total_district_customers,
                    "workload": total_district_workload,
                    "demand": total_district_demand,
                },
                "Customers Infeasibility": infeasibility_customers,
                "Demand Infeasibility": infeasibility_demand,
                "Workload Infeasibility": infeasibility_workload,
                "Total Infeasibility": infeasibility_total,
            },
        )


class PR:
    def __init__(self, tdp_instance, i_max, elite_set_length):
        self.tdpInstance = tdp_instance
        self.elite_set_length = elite_set_length
        self.i_max = i_max
        self.eliteSolutions = []
        for i in range(self.elite_set_length):
            self.tdpInstance.centers_depots = self.tdpInstance.select_centroids()
            incumbent_objective, incumbent_infeasibility, incumbent_solution = (
                self.tdpInstance.constructDistricts()[:3]
            )

            improved_objective, improved_infeasibility, improved_solution = (
                self.tdpInstance.localSearch(
                    incumbent_objective, incumbent_infeasibility, incumbent_solution
                )
            )
            current_merit = (
                self.tdpInstance.llambda
                * (improved_objective / self.tdpInstance.graph_diameter)
                + (1 - self.tdpInstance.llambda) * improved_infeasibility
            )

            solution = {
                "Objective": improved_objective,
                "Infeasibility": improved_infeasibility,
                "Merit": current_merit,
                "Centers": {
                    i: self.tdpInstance.get_district_center(improved_solution[i])
                    for i in improved_solution
                },
                "Districts": improved_solution,
            }

            index = bisect.bisect_left(
                [x["Objective"] for x in self.eliteSolutions], improved_objective
            )
            self.eliteSolutions.insert(index, solution)

        self.bestSolution = cp.deepcopy(self.eliteSolutions[0])

    def distanceCalculation(self, solution1, solution2):
        # Align districts via Hungarian assignment so set comparisons share a consistent ordering.
        distance_dict = self.tdpInstance.shortest_paths_dict

        # Create the cost matrix
        cost_matrix = np.full(
            (len(solution1["Centers"]), len(solution2["Centers"])), np.inf
        )

        # Fill the cost matrix with distance values from shortest_paths_dict
        for i, cntr1 in solution1["Centers"].items():
            for j, cntr2 in solution2["Centers"].items():
                cost_matrix[i, j] = distance_dict[cntr1][cntr2]

        # Apply the Hungarian algorithm
        dist1_ind, dist2_ind = linear_sum_assignment(cost_matrix)

        nodes_assigned_different_district = 0

        for dist1, dist2 in zip(dist1_ind, dist2_ind):
            nodes_assigned_different_district += sum(
                node not in solution2["Districts"][dist2]
                for node in solution1["Districts"][dist1]
            )

        fraction_assigned_different_district = nodes_assigned_different_district / len(
            self.tdpInstance.nodes
        )

        return list(zip(dist1_ind, dist2_ind)), fraction_assigned_different_district

    def distanceBetweenSolutions(self, current_solution, solutionsList):
        # Average pairwise fraction of nodes moved, used as a diversity proxy.
        total_distance = 0
        for sol in solutionsList:
            districts_matching, fractional_difference = self.distanceCalculation(
                current_solution, sol
            )
            total_distance += fractional_difference

        average_distance = (1 / len(solutionsList)) * total_distance

        return average_distance

    def generateSolutions(self):
        # Build new incumbents, run local search, and keep those that improve merit/diversity.
        for i in range(self.i_max):
            self.tdpInstance.centers_depots = self.tdpInstance.select_centroids()

            incumbent_objective, incumbent_infeasibility, incumbent_districts = (
                self.tdpInstance.constructDistricts()[:3]
            )
            improved_objective, improved_infeasibility, improved_districts = (
                self.tdpInstance.localSearch(
                    incumbent_objective, incumbent_infeasibility, incumbent_districts
                )
            )

            current_merit = (
                self.tdpInstance.llambda
                * (improved_objective / self.tdpInstance.graph_diameter)
                + (1 - self.tdpInstance.llambda) * improved_infeasibility
            )

            improved_solution = {
                "Objective": improved_objective,
                "Infeasibility": improved_infeasibility,
                "Merit": current_merit,
                "Centers": {
                    i: self.tdpInstance.get_district_center(improved_districts[i])
                    for i in improved_districts
                },
                "Districts": improved_districts,
            }

            distance_between_elite_solutions = self.distanceBetweenSolutions(
                improved_solution, self.eliteSolutions
            )
            # print("Distance between elite solutions is: ", distance_between_elite_solutions)
            if current_merit < self.eliteSolutions[0]["Merit"] or (
                current_merit < self.eliteSolutions[-1]["Merit"]
                and distance_between_elite_solutions > 0.6
            ):
                self.eliteSolutions.remove(self.eliteSolutions[-1])
                index = bisect.bisect_left(
                    [x["Objective"] for x in self.eliteSolutions], improved_objective
                )
                self.eliteSolutions.insert(index, improved_solution)

    def prEvaluation(self, elite_solution1, elite_solution2):
        # Quick comparison routine that keeps the better parent to seed recombination.

        solution_infeasibility1 = sum(
            self.tdpInstance.get_district_infeasibility(elite_solution1[k])[0]
            for k in elite_solution1
        )
        solution_infeasibility2 = sum(
            self.tdpInstance.get_district_infeasibility(elite_solution2[k])[0]
            for k in elite_solution2
        )

        solution1_objective = max(
            self.tdpInstance.get_district_diameter_numpy(dist)["diameter"]
            for dist in elite_solution1.values()
        )
        solution2_objective = max(
            self.tdpInstance.get_district_diameter_numpy(dist)["diameter"]
            for dist in elite_solution2.values()
        )

        solution1_merit = (
            self.tdpInstance.llambda
            * (solution1_objective / self.tdpInstance.graph_diameter)
            + (1 - self.tdpInstance.llambda) * solution_infeasibility1
        )

        solution2_merit = (
            self.tdpInstance.llambda
            * (solution2_objective / self.tdpInstance.graph_diameter)
            + (1 - self.tdpInstance.llambda) * solution_infeasibility2
        )

        if solution1_merit > solution2_merit:
            best_merit = solution2_merit
            solution_objective = solution2_objective
            best_solution = elite_solution2
            solution_infeasibility = solution_infeasibility2
        else:
            best_merit = solution1_merit
            solution_objective = solution1_objective
            best_solution = elite_solution1
            solution_infeasibility = solution_infeasibility1

        weight_district_best = best_merit

        return (
            weight_district_best,
            solution_objective,
            solution_infeasibility,
            best_solution,
        )

    def performPR(self):
        logging.debug(
            f"Starting PR -  merit: {self.bestSolution['Merit']}, objecitve: {self.bestSolution['Objective']}, infeasibility {self.bestSolution['Infeasibility']}"
        )

        # Explore every elite pair and attempt to recombine their complementary regions.
        for i, j in itertools.combinations(range(self.elite_set_length), 2):

            district_matching, fraction_of_nodes = self.distanceCalculation(
                self.eliteSolutions[i], self.eliteSolutions[j]
            )

            intermediary_solution1 = cp.deepcopy(self.eliteSolutions[i]["Districts"])
            intermediary_solution2 = cp.deepcopy(self.eliteSolutions[j]["Districts"])

            for dist1, dist2 in district_matching:
                # Build move lists describing how each parent assigns nodes differently.
                nodes_to_move1 = {}
                nodes_to_move2 = {}
                for node in intermediary_solution1[dist1]:
                    if node not in intermediary_solution2[dist2]:
                        nodes_to_move1[node] = {"from": dist1, "to": dist2}
                for node in intermediary_solution2[dist2]:
                    if node not in intermediary_solution1[dist1]:
                        nodes_to_move2[node] = {"from": dist2, "to": dist1}

            incumbent_merit = float("inf")
            incumbent_solution = None
            incumbent_objective = float("inf")
            incumbent_infeasibility = float("inf")

            current_diameters = {
                k: self.tdpInstance.get_district_diameter_numpy(
                    intermediary_solution1[k]
                )
                for k in intermediary_solution1
            }
            current_infeasibilities = {
                k: self.tdpInstance.get_district_infeasibility(
                    intermediary_solution1[k]
                )[0]
                for k in intermediary_solution1
            }
            # find the best improvement from all intermediary solutions from i to j
            for nd, move in nodes_to_move1.items():
                # update solution
                intermediary_solution1[move["from"]].remove(nd)
                intermediary_solution1[move["to"]].append(nd)

                # evaluate solution
                infeasibility = sum(
                    current_infeasibilities[k]
                    for k in intermediary_solution1
                    if k not in move.values()
                )
                infeasibility += sum(
                    self.tdpInstance.get_district_infeasibility(
                        intermediary_solution1[k]
                    )[0]
                    for k in intermediary_solution1
                    if k in move.values()
                )

                objective = max(
                    current_diameters[k]["diameter"]
                    for k in intermediary_solution1
                    if k not in move.values()
                )
                objective = max(
                    objective,
                    max(
                        self.tdpInstance.get_district_diameter_numpy(
                            intermediary_solution1[k]
                        )["diameter"]
                        for k in intermediary_solution1
                        if k in move.values()
                    ),
                )

                merit = (
                    self.tdpInstance.llambda
                    * (objective / self.tdpInstance.graph_diameter)
                    + (1 - self.tdpInstance.llambda) * infeasibility
                )

                if merit < incumbent_merit:
                    incumbent_solution = cp.deepcopy(intermediary_solution1)
                    incumbent_merit = merit
                    incumbent_objective = objective
                    incumbent_infeasibility = infeasibility

            current_diameters = {
                k: self.tdpInstance.get_district_diameter_numpy(
                    intermediary_solution2[k]
                )
                for k in intermediary_solution1
            }
            current_infeasibilities = {
                k: self.tdpInstance.get_district_infeasibility(
                    intermediary_solution2[k]
                )[0]
                for k in intermediary_solution1
            }
            # find the best improvement from all intermediary solutions from j to i
            for nd, move in nodes_to_move2.items():
                # update solution
                intermediary_solution2[move["from"]].remove(nd)
                intermediary_solution2[move["to"]].append(nd)

                # evaluate solution
                infeasibility = sum(
                    current_infeasibilities[k]
                    for k in intermediary_solution2
                    if k not in move.values()
                )
                infeasibility += sum(
                    self.tdpInstance.get_district_infeasibility(
                        intermediary_solution2[k]
                    )[0]
                    for k in intermediary_solution2
                    if k in move.values()
                )

                # objective = max(self.tdpInstance.get_district_diameter_numpy(dist)['diameter'] for dist in intermediary_solution2.values())
                objective = max(
                    current_diameters[k]["diameter"]
                    for k in intermediary_solution2
                    if k not in move.values()
                )
                objective = max(
                    objective,
                    max(
                        self.tdpInstance.get_district_diameter_numpy(
                            intermediary_solution2[k]
                        )["diameter"]
                        for k in intermediary_solution2
                        if k in move.values()
                    ),
                )

                merit = (
                    self.tdpInstance.llambda
                    * (objective / self.tdpInstance.graph_diameter)
                    + (1 - self.tdpInstance.llambda) * infeasibility
                )

                if merit < incumbent_merit:
                    incumbent_solution = cp.deepcopy(intermediary_solution2)
                    incumbent_merit = merit
                    incumbent_objective = objective
                    incumbent_infeasibility = infeasibility

            if incumbent_solution:
                improved_objective, improved_infeasibility, improved_districts = (
                    self.tdpInstance.localSearch(
                        incumbent_objective, incumbent_infeasibility, incumbent_solution
                    )
                )

            new_merit = (
                self.tdpInstance.llambda
                * (improved_objective / self.tdpInstance.graph_diameter)
                + (1 - self.tdpInstance.llambda) * improved_infeasibility
            )

            if new_merit < self.bestSolution["Merit"]:
                self.bestSolution = {
                    "Objective": improved_objective,
                    "Infeasibility": improved_infeasibility,
                    "Merit": new_merit,
                    "Centers": {
                        i: self.tdpInstance.get_district_center(improved_districts[i])
                        for i in improved_districts
                    },
                    "Districts": improved_districts,
                }
                logging.debug(
                    f"PR improvement -  merit: {self.bestSolution['Merit']}, objecitve: {self.bestSolution['Objective']}, infeasibility {self.bestSolution['Infeasibility']}"
                )

        logging.debug(
            f"PR finished - {self.bestSolution['Objective']}, {self.bestSolution['Infeasibility']}"
        )


class BVNS:
    """
    Class implementing the Basic Variable Neighborhood Search algorithm for solving the TDP (Traveling District Problem).

    Parameters:
    -----------
    tdp_instance : TDP instance object
        An instance of the TDP problem.
    shaking_steps : int
        The maximum number of shaking steps allowed in the BVNS algorithm.
    fail_max : int
        The maximum number of consecutive failures allowed in the BVNS algorithm.

    Attributes:
    -----------
    tdpInstance : TDP instance object
        An instance of the TDP problem.
    incumbent_objective : float
        The objective value of the incumbent solution.
    incumbent_infeasibility : float
        The infeasibility value of the incumbent solution.
    incumbent_solution : dictionary
        The incumbent solution in the form of a dictionary where keys are depot indices and values
        are lists of node indices.
    current_merit : float
        The current merit value of the solution.
    shakeSteps : int
        The maximum number of shaking steps allowed in the BVNS algorithm.

    maxFails : int
        The maximum number of total failures allowed in the BVNS algorithm.
    """

    def __init__(
        self,
        tdp_instance,
        shaking_steps,
        fail_max,
        nrInitSolutions=100,
        startingLambda=0.42,
        lambdaRange=[0.1, 0.4, 0.7, 1],
    ):

        self.tdpInstance = tdp_instance
        self.incumbent_objective = float("inf")
        self.incumbent_infeasibility = float("inf")
        self.current_merit = float("inf")
        self.current_lambda = startingLambda
        self.agent = RLAgent(state_dim=7, action_dim=4)

        for i in range(nrInitSolutions):
            # Multi-start construction: each run seeds BVNS with a different centroid selection.
            self.tdpInstance.centers_depots = self.tdpInstance.select_centroids()
            objective, infeasibility, districts = self.tdpInstance.constructDistricts()[
                :3
            ]
            merit = (
                objective / self.tdpInstance.graph_diameter
            ) + self.current_lambda * infeasibility

            if objective < self.incumbent_objective:
                self.incumbent_objective = objective
                self.incumbent_infeasibility = infeasibility
                self.incumbent_solution = cp.deepcopy(districts)
                self.current_merit = merit

        self.current_merit = (
            self.incumbent_objective / self.tdpInstance.graph_diameter
        ) + self.current_lambda * self.incumbent_infeasibility
        self.shakeSteps = shaking_steps
        self.maxFails = fail_max
        self.lambdaList = lambdaRange

        # Cache the frontier edges per district to accelerate adjacency-based moves.
        self.adjEdges = {
            k: {
                (o, d)
                for o in self.incumbent_solution[k]
                for d in self.tdpInstance.graph_input.adj[o]
                if not d in self.incumbent_solution[k]
            }
            for k in self.incumbent_solution
        }

        self.adjacentDistricts = {}
        # Map each district to the neighboring districts via nodes that touch across borders.

        self.node_district_matching = {}
        for dist_idx in self.incumbent_solution:
            for i in self.incumbent_solution[dist_idx]:
                self.node_district_matching[i] = dist_idx

        for o, d in self.tdpInstance.graph_input.edges:
            dist1 = self.node_district_matching[o]
            dist2 = self.node_district_matching[d]

            if dist1 != dist2:
                if dist1 not in self.adjacentDistricts:
                    self.adjacentDistricts[dist1] = {}
                if dist2 not in self.adjacentDistricts:
                    self.adjacentDistricts[dist2] = {}

                if dist2 not in self.adjacentDistricts[dist1]:
                    self.adjacentDistricts[dist1][dist2] = set()
                if dist1 not in self.adjacentDistricts[dist2]:
                    self.adjacentDistricts[dist2][dist1] = set()

                self.adjacentDistricts[dist1][dist2].add(o)
                self.adjacentDistricts[dist2][dist1].add(d)

    def calculateProbabilities(self, solutionToShake, neighborhood):

        accepted_districts = []
        testLengthDict = {}

        for i in solutionToShake:
            testLengthDict[i] = len(solutionToShake[i])
            if testLengthDict[i] > neighborhood:
                accepted_districts.append(i)

        infeasibilities = {}
        total_activtities = {}
        for k in solutionToShake:
            infeasibilities[k], total_activtities[k] = (
                self.tdpInstance.get_district_infeasibility(solutionToShake[k])
            )

        total_infeasibility = sum(infeasibilities[i] for i in accepted_districts)

        probabilities = {}
        if total_infeasibility:
            # Prefer districts that currently violate balance the most.
            for cluster, infeas in infeasibilities.items():
                if cluster in accepted_districts:
                    probabilities[cluster] = infeas / total_infeasibility

            chosen_district = random.choices(
                list(probabilities.keys()), list(probabilities.values())
            )[0]

        else:
            # If everything is feasible, pick randomly to keep shaking unbiased.
            chosen_district = random.choice(accepted_districts)
            logging.debug(
                f"chosen district: {chosen_district}, lambda: {self.tdpInstance.llambda}"
            )

        return chosen_district

    def extract_state(self, objective, infeasibility, k, failure):
        sizes = [len(v) for v in self.incumbent_solution.values()]
        n_nodes = len(self.tdpInstance.nodes)

        state = np.array(
            [
                objective / self.tdpInstance.graph_diameter,
                infeasibility,
                k / self.shakeSteps,
                np.mean(sizes) / n_nodes,
                np.std(sizes) / n_nodes,
                self.tdpInstance.llambda,
                failure / self.maxFails,
            ],
            dtype=np.float32,
        )

        return state

    def move_cluster_between_districts(
        self,
        node_district_matching,
        districts,
        adjacentDistricts,
        border_edges,
        depot_from,
        num_nodes=1,
    ):

        # Initialize the cluster, select starting node randomly, initialize sorted distance list

        candidate_cluster = []
        depot_to = random.choice(list(adjacentDistricts[depot_from].keys()))
        starting_random_node = random.choice(
            list(adjacentDistricts[depot_from][depot_to])
        )
        sorted_nodes = sorted(
            (node for node in self.tdpInstance.nodes if node in districts[depot_from]),
            key=lambda x: self.tdpInstance.shortest_paths_dict[starting_random_node][x],
        )
        candidate_cluster = sorted_nodes[:num_nodes]

        # Update relevant dictionaries
        for node_to_move in candidate_cluster:
            districts, node_district_matching, adjacentDistricts = (
                self.tdpInstance.moveNode(
                    districts,
                    node_to_move,
                    depot_from,
                    depot_to,
                    node_district_matching,
                    adjacentDistricts,
                    border_edges,
                )
            )

        return districts, adjacentDistricts, border_edges

    def clusterShake(
        self, solutionToShake, adjacentDistricts, border_edges, neighborhood
    ):

        # Perform the shaking step by moving a handful of connected nodes from a chosen district.
        shaking_input = cp.deepcopy(solutionToShake)
        node_district_matching = {}

        for k in shaking_input:
            for i in shaking_input[k]:
                node_district_matching[i] = k

        depot_from = self.calculateProbabilities(solutionToShake, neighborhood)

        while len(adjacentDistricts[depot_from]) == 0:
            # Reselect if the chosen district no longer has borders to exchange.
            depot_from = self.calculateProbabilities(solutionToShake, neighborhood)

        shaking_input, adjacentDistricts, border_edges = (
            self.move_cluster_between_districts(
                node_district_matching,
                shaking_input,
                adjacentDistricts,
                border_edges,
                depot_from,
                neighborhood,
            )
        )

        shaking_objective = max(
            self.tdpInstance.get_district_diameter_numpy(dist)["diameter"]
            for dist in shaking_input.values()
        )

        shaking_infeasibility = 0
        activity_totals = {}
        for k in shaking_input:
            dist_infeasibility, activity_totals[k] = (
                self.tdpInstance.get_district_infeasibility(shaking_input[k])
            )
            shaking_infeasibility += dist_infeasibility

        return (
            shaking_input,
            shaking_objective,
            shaking_infeasibility,
            activity_totals,
            adjacentDistricts,
            border_edges,
        )

    def performBVNS(self, train_agent=True):

        solutions_collection = []
        objective_collection = []
        infeasibility_collection = []

        incumbent_solution = cp.deepcopy(self.incumbent_solution)
        best_solution = incumbent_solution

        best_objective = self.incumbent_objective
        best_infeasibility = self.incumbent_infeasibility

        objective_collection.append(best_objective)
        infeasibility_collection.append(best_infeasibility)

        current_merit = self.current_merit

        incumbent_adjacentDistricts = cp.deepcopy(self.adjacentDistricts)

        border_edges = {
            dist: {
                (o, d)
                for o in incumbent_solution[dist]
                for d in self.tdpInstance.graph_input.adj[o]
                if d not in incumbent_solution[dist]
            }
            for dist in incumbent_solution
        }

        solutions_collection.append(
            {
                "objective": best_objective,
                "infeasibility": best_infeasibility,
                "merit": current_merit,
                "lambda": self.current_lambda,
                "time": time.time(),
            }
        )

        failure = 0
        k = 1

        while failure < self.maxFails:

            # ===== STATE =====
            state = self.extract_state(best_objective, best_infeasibility, k, failure)

            # ===== ACTION =====
            action = self.agent.act(state)

            # ===== APPLY ACTION =====
            if action == 0:
                k = min(self.shakeSteps, k + 1)

            elif action == 1:
                k = max(1, k - 1)

            elif action == 2:
                self.tdpInstance.llambda = random.choice(self.lambdaList)

            elif action == 3:
                k = random.randint(1, self.shakeSteps)

            # Recompute current_merit với lambda hiện tại (có thể đã thay đổi bởi action 2)
            current_merit = (
                best_objective / self.tdpInstance.graph_diameter
            ) + self.tdpInstance.llambda * best_infeasibility

            # ===== SHAKE =====
            (
                shaking_input,
                shaking_objective,
                shaking_infeasible,
                activity_totals,
                incumbent_adjacentDistricts,
                border_edges,
            ) = self.clusterShake(
                incumbent_solution, incumbent_adjacentDistricts, border_edges, k
            )

            # ===== LOCAL SEARCH =====
            (
                local_objective,
                local_infeasibility,
                local_solution,
                incumbent_adjacentDistricts,
                border_edges,
            ) = self.tdpInstance.nodeBestImprovement(
                shaking_input,
                shaking_objective,
                shaking_infeasible,
                incumbent_adjacentDistricts,
                border_edges,
            )

            # ===== EVALUATE =====
            new_merit = (
                local_objective / self.tdpInstance.graph_diameter
            ) + self.tdpInstance.llambda * local_infeasibility

            # ===== REWARD =====
            # cả hai merit dùng cùng lambda → so sánh hợp lệ
            reward = current_merit - new_merit

            new_state = self.extract_state(
                local_objective, local_infeasibility, k, failure
            )

            # ===== ACCEPT =====
            if new_merit < current_merit:

                best_solution = cp.deepcopy(local_solution)
                best_objective = local_objective
                best_infeasibility = local_infeasibility

                incumbent_solution = cp.deepcopy(best_solution)
                current_merit = new_merit

                # reset neighborhood
                k = 1
                failure = 0
                done = False

                objective_collection.append(best_objective)
                infeasibility_collection.append(best_infeasibility)

                solutions_collection.append(
                    {
                        "objective": best_objective,
                        "infeasibility": best_infeasibility,
                        "merit": current_merit,
                        "lambda": self.tdpInstance.llambda,
                        "time": time.time(),
                    }
                )

            else:
                failure += 1
                done = failure >= self.maxFails

            # ===== RL UPDATE =====
            if train_agent:
                self.agent.store(state, action, reward, new_state, done)
                self.agent.train()

        return (
            objective_collection,
            infeasibility_collection,
            best_solution,
            solutions_collection,
        )


import torch
import torch.nn as nn
import torch.optim as optim
import random


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.target.load_state_dict(self.model.state_dict())
        self.update_counter = 0
        self.update_target_every = 200

    def save(self, path="rl_model.pt"):
        torch.save(
            {
                "model": self.model.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_counter": self.update_counter,
            },
            path,
        )

    def load(self, path="rl_model.pt"):
        import os

        if os.path.exists(path):
            ckpt = torch.load(path, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.target.load_state_dict(ckpt["target"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.update_counter = ckpt["update_counter"]

    def act(self, state, eps=0.1):
        if random.random() < eps:
            return random.randint(0, 3)
        with torch.no_grad():
            q = self.model(torch.FloatTensor(state))
        return torch.argmax(q).item()

    def store(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, done = zip(*batch)

        s = torch.FloatTensor(s)
        s2 = torch.FloatTensor(s2)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        done = torch.FloatTensor(done)

        # Q(s,a)
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()

        # Q target — không bootstrap từ terminal state
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]

        target = r + self.gamma * q_next * (1 - done)

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target.load_state_dict(self.model.state_dict())
