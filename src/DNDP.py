import numpy as np
import pandas as pd
import logging
import scipy.optimize as opt
import scipy.sparse.csgraph as csgraph

from src.network import Node, Link, Network
from log import setup_log

logger = setup_log(log_dir="../")

class DNDP:
    def __init__(self, work_dir, node_file, link_file, OD_file):
        # Unchangeable network attributes
        self.node_df = pd.read_csv(work_dir + node_file)
        self.link_df = pd.read_csv(work_dir + link_file)
        self.OD = np.load(work_dir + OD_file)

        self.nodes = {}
        self.links = {}
        self.network = None

        self.w_1 = 100  # weight for u_ij in objective
        self.w_2 = 1  # weight for z_ij in objective
        self.w_3 = 1  # weight for SO cost in objective
        self.LB_record = []
        self.UB_record = []
        self.A_0 = set() # set of links with y_ij = 0
        self.A_1 = set() # set of links with y_ij = 1
        self.A_frac = set()
        self.best_solution = None
        self.base_model = None
        self.RMP = None
        self.solver = None
        self.dndp_epsilon = 1e-2

    def run_workflow(self):
        self.construct_network()
        self.build_model()
        self.solve_model()

    def construct_network(self):
        """
        Construct the network from node and link files.
        :return: None
        """
        self.construct_nodes()
        self.construct_links()
        self.read_OD_data()
        self.network = Network(self.nodes, self.links, self.link_df)

    def build_model(self):
        pass

    def solve_model(self):
        pass

    def construct_nodes(self):
        # Read the node data
        nodes_id = self.node_df["ID"].to_numpy()
        for id, node in enumerate(nodes_id):
            self.nodes[id + 1] = Node(node)

    def construct_links(self):
        # Read the link data
        self.link_df[["init_node", "term_node", "power", "parking_num"]] = self.link_df[
            ["init_node", "term_node", "power", "parking_num"]
        ].astype(int)
        self.link_df[["capacity", "length", "free_flow_time", "b"]] = self.link_df[
            ["capacity", "length", "free_flow_time", "b"]
        ].astype(float)
        self.link_df["length"] = self.link_df["length"] * 1000  # Convert km to m
        for idx, link in self.link_df.iterrows():
            self.links[(link.init_node, link.term_node)] = Link(link)

    def read_OD_data(self):
        # Already read when initializing
        pass

    def build_base_model(self):
        pass

    def solve_UE_TAP(self, epsilon, maxIter):
        """
        Solve the relaxed NLP subproblem, namely the User Equilibrium Traffic Assignment Problem (UE-TAP).
        return the subproblem variable x_ij(link flow)
        use x_ij to calculate system optimal objective value, as an over-estimator upper bound of the original NLP subproblem.
        """

        def determine_search_direction():
            """
            # Step 2: Determine search direction

            2.1 Gradient at current solution x_k is simply the BPR function evaluated at x_k,
            namely: ∇f(x_k) = t_a(x_k).
            2.2 The LP to solve is min <s_k, ∇f(x_k)>, s.t. s_k ∈ feasible region.
            The solution of LP min <s_k, ∇f(x_k)> is actually the All or Nothing assignment result,
            because its physical meaning is to find the route that minimizes the total travel cost
            given the fixed link cost.
            """

            def Dijkstra():
                """
                Use SciPy's Dijkstra algorithm to find shortest paths from an origin.
                """
                num_nodes = len(self.nodes)
                adj_matrix = np.full((num_nodes, num_nodes), np.inf)
                # todo: change link.time_cost after each iteration
                for idx, (tail, head) in enumerate(self.network.links_id):
                    adj_matrix[tail - 1, head - 1] = self.network.link_time_cost[idx]

                # Run Dijkstra for a single source
                dists, preds = csgraph.dijkstra(
                    csgraph=adj_matrix, directed=True, return_predecessors=True
                )

                return dists, preds

            def Find_Preds(pred, Dest):
                """
                Trace the predecessors of a node to find the shortest path.
                """
                # pred[Dest-1] is the predecessor of Dest, namely: Dest is the head node of link, pred[Dest-1] is the tail node of link. Dest-1 because of 0-based indexing.
                while pred[Dest] != -9999:
                    yield (pred[Dest], Dest)
                    Dest = pred[Dest]

            """
            Start of shortest path algorithm
            """
            Auxiliary_Flow = {link: 0.0 for link in self.network.links_id}
            SPTT = 0.0
            nodes_id = range(len(self.nodes))
            labels, preds = Dijkstra()
            preds = preds.tolist()
            for Origin in nodes_id:
                pred = preds[Origin]
                for Dest in nodes_id:
                    demand = self.OD[Origin][Dest]  # adjust for 0-based indexing
                    if Origin == Dest or demand == 0 or labels[Origin, Dest] == np.inf:
                        continue
                    for tail, head in Find_Preds(pred, Dest):
                        Auxiliary_Flow[tail + 1, head + 1] += demand
                    SPTT += labels[Origin, Dest] * demand
            s_k = Auxiliary_Flow
            return SPTT, s_k

        def determine_step_size(s_k):
            """
            Step 3: Determine step size
            To determine step_size that
            Min: f(x_k + step_size * (s_k - x_k)) w.r.t step_size
            ∇f(step_size) = <∇f(x_k + step_size * (s_k - x_k)), (s_k - x_k)> = 0
            in which x_k is the current flow, (s_k - x_k) is search direction.
            This is a LP w.r.t step_size, easy to solve.
            """
            alpha = self.network.link_b[0]
            beta = self.network.link_power[0]
            s_k = np.array(list(s_k.values()))

            def step_size(step_size):
                BPR = self.network.link_fft * (
                    1
                    + alpha
                    * (
                        (
                            self.network.link_flow
                            + step_size * (s_k - self.network.link_flow)
                        )
                        / self.network.link_capacity
                    )
                    ** beta
                )
                sum_derivative = np.sum(BPR * (s_k - self.network.link_flow))
                return sum_derivative

            sol = opt.fsolve(step_size, 0.01)[0]
            step_size = np.clip(sol, 0.0, 1.0)
            return step_size

        """
        Start of the Frank-Wolfe algorithm
        """
        # 1. Initialize solution
        iter = 0
        gap = float("inf")
        alpha = self.network.link_b
        beta = self.network.link_power

        while gap >= epsilon:
            if self.network.link_flow is None:
                self.network.link_flow = np.zeros(len(self.links))
            else:
                pass

            # 2. Calculate CURRENT TSTT (before finding new direction)
            TSTT = np.sum(
                self.network.link_flow * self.network.link_time_cost
            )  # Total System Travel Time

            # 3. Determine search direction
            SPTT, s_k = determine_search_direction()

            # 4. calculate convergence
            if iter > 0:
                gap = round(abs(1 - SPTT / TSTT), 5)

            if gap < epsilon:
                logging.info(
                    f"UE-TAP converged at iteration: {iter} , current gap: {gap}"
                )
                break
            if iter >= maxIter:
                logging.info(f"UE-TAP did not converge, current gap:{gap}")
                break

            # 5. Determine step size
            step_size = determine_step_size(s_k)
            # AON for first iteration to obtain an initial feasible solution
            if iter == 0 and gap > 0.2:
                step_size = 1

            # 6. New iteration point
            s_k_value = np.array(list(s_k.values()))
            self.network.link_flow = self.network.link_flow + step_size * (
                s_k_value - self.network.link_flow
            )
            self.network.link_time_cost = self.network.link_fft * (
                1
                + alpha * (self.network.link_flow / self.network.link_capacity) ** beta
            )

            iter += 1

        return TSTT