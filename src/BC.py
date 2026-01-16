"""
Pseudo Code for Outer Approximation (OA) Algorithm:
0. Initialize: LB = -infinity,  UB = +infinity

while gap < \epsilon
1. Initialize a feasible y_k on the upper-level
2. solve the NLP, upper-level is minimize SO+c_j* y_j, lower-level is UE-TAP(non-convex due to KKT constraints). Simplifiy the Non-convex non-linear programming as a simple UE-TAP, obtain x_*. Use x* to calculate upper-level objective, obtain and update upper-bound UB_k.
3. Add optimality cut and generalized benders decomposition cut, substitute non-linear f(x) and g(x)≤0, respectively. Solve the MILP w.r.t. y, obtain and update LB.
4. Add Integer cut(to avoid duplicate enumeration of y_k)
5. If UB - LB < \epsilon, terminate, else return to 1.
"""

import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import itertools
import pyomo.environ as pyo
import traceback
import time
import logging

from src.DNDP import DNDP
from contextlib import redirect_stdout, redirect_stderr
from pyomo.core import TransformationFactory
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from log import setup_log

logger = setup_log(log_dir="../")

class BC(DNDP):
    """
    subclass of DNDP, implement the Branch-and-Cut algorithm for solving the DNDP problem.
    Inherited methods:
    run_workflow(),
    construct_network(), construct_nodes(), construct_links(), read_OD_data(): construct the network

    Overrided methods:
    build_model(): build the base MINLP model
    solve_model(): implement the Branch-and-Cut algorithm
    """
    def __init__(self, work_dir="", node_file="", link_file="", OD_file=""):
        super().__init__(work_dir, node_file, link_file, OD_file)

    def build_model(self):
        self.minlp = MINLP()
        self.minlp.model = self.build_base_model()
        self.minlp.build_milp()

    def solve_model(self):
        LB, UB = -float("inf"), float("inf")
        gap = float("inf")

        # 1. Initialize a feasible y_iter on the upper-level
        k = 1
        while gap > self.dndp_epsilon:
            logging.info(
                f"############### DNDP Iteration {k} starts, current gap: {gap}, LB: {LB}, UB: {UB} ###############"
            )
            # 2. Solve the Master MINLP problem, obtain y^k and LB^k, update LB.
            self.minlp.solve_minlp()

            ## IF infeasible, terminate, return best solution found so far.
            temp_LB = self.w_1 * pyo.value(
                sum(self.minlp.model.u_ij[i, j] for i, j in self.minlp.model.A)
            ) + self.w_2 * pyo.value(
                sum(self.minlp.model.z_ij[i, j] for i, j in self.minlp.model.A)
            ) + self.w_3 * self.minlp.true_obj_cost

            if temp_LB < LB * (1 - self.dndp_epsilon):
                logging.warning("!!!!MINLP solver returns a worse LB, this indicates an error with ue-reduction cut or no-good cut!!!!")
            LB = max(temp_LB, LB)

            if LB >= UB * (1 - self.dndp_epsilon):
                self.LB_record.append(LB)
                self.UB_record.append(UB)
                logging.info(
                    f"############### DNDP Iteration {k} ends, LB={LB} >= UB*(1-epsilon)={UB*(1-self.dndp_epsilon)} ############### \n"
                    f"~~~~~~~~~~~~~DNDP converged at iteration: {k}, reach goal gap: {self.dndp_epsilon}~~~~~~~~~~~~~"
                )
                break

            # 3. Solve the UE subproblem with fixed y^k, obtain x^k and UB^k, update UB
            self.network.link_capacity = np.array(
                [self.minlp.model.capacity[i, j].value for i, j in self.minlp.model.A]
            )
            self.solve_UE_TAP(epsilon=1e-4, maxIter=1000)

            temp_UB = (
                self.w_1
                * pyo.value(
                    sum(self.minlp.model.u_ij[i, j] for i, j in self.minlp.model.A)
                )
                + self.w_2
                * pyo.value(
                    sum(self.minlp.model.z_ij[i, j] for i, j in self.minlp.model.A)
                )
                + self.w_3 * sum(self.network.link_flow * self.network.link_time_cost)
            )
            if temp_UB < UB:
                UB = temp_UB
                self.best_solution = [
                    pyo.value(self.minlp.model.y_ij[i, j])
                    for i, j in self.minlp.model.A
                ]

            # 4. Add no good integer cut and UE-reduction cut
            self.add_no_good_cut()
            self.add_UE_reduction_cut()

            # 5. if (UB - LB) / UB < \epsilon, terminate, else return to 2.
            self.LB_record.append(LB)
            self.UB_record.append(UB)

            if LB >= UB * (1 - self.dndp_epsilon):
                self.LB_record.append(LB)
                logging.info(
                    f"############### DNDP Iteration {k} ends, LB={LB} >= UB*(1-epsilon)={UB*(1-self.dndp_epsilon)} ############### \n"
                    f"~~~~~~~~~~~~~DNDP converged at iteration: {k}, reach goal gap: {self.dndp_epsilon}~~~~~~~~~~~~~ \n"
                    f"Best solution saved to best_solution.csv, "
                )
                break

            gap = round(abs(1 - LB / UB), 5)
            logging.info(
                f"############### DNDP Iteration {k} ends, current gap: {gap}, LB: {LB}, UB: {UB} ###############"
            )

            k += 1

        ## Save best solution and LB and UB record
        np.savetxt("best_solution.csv", np.array(self.best_solution), delimiter=",")
        convergence_record = pd.DataFrame(
            {"LB": self.LB_record, "UB": self.UB_record}
        )
        convergence_record.to_csv("convergence_record.csv", index=False)

    def build_base_model(self):
        model = pyo.ConcreteModel()

        # 1. Construct sets
        ## Node set
        rs = list(itertools.product(list(self.nodes.keys()), repeat=2))
        model.R = pyo.Set(initialize=self.nodes.keys())
        model.S = pyo.Set(initialize=self.nodes.keys())
        model.RS = pyo.Set(initialize=rs)

        ## Link set
        model.A = pyo.Set(initialize=[link for link in self.links.keys()])

        # 2. Construct parameters

        # 2.1 Network parameters
        model.l_ij = pyo.Param(
            model.A,
            within=pyo.NonNegativeReals,
            initialize={
                id: length
                for id, length in zip(self.network.links_id, self.network.link_length)
            },
        )
        model.c_0 = pyo.Param(
            model.A,
            within=pyo.NonNegativeReals,
            initialize={
                id: c_0 for id, c_0 in zip(self.network.links_id, self.network.link_c_0)
            },
        )
        model.capacity = pyo.Param(
            model.A,
            within=pyo.NonNegativeReals,
            initialize={
                id: capacity
                for id, capacity in zip(
                    self.network.links_id, self.network.link_capacity
                )
            },
            mutable=True,
        )
        model.fft = pyo.Param(
            model.A,
            within=pyo.NonNegativeReals,
            initialize={
                id: link_fft
                for id, link_fft in zip(self.network.links_id, self.network.link_fft)
            },
        )
        model._alpha = 0.15
        model._beta = 4
        model.lane_num = pyo.Param(
            model.A,
            within=pyo.NonNegativeIntegers,
            initialize={
                id: link_lane_num
                for id, link_lane_num in zip(
                    self.network.links_id, self.network.link_lane_num
                )
            },
        )
        model.road_width = pyo.Param(
            model.A,
            within=pyo.NonNegativeReals,
            initialize={
                id: road_width
                for id, road_width in zip(
                    self.network.links_id, self.network.link_road_width
                )
            },
        )

        model.lane_separator = pyo.Param(
            model.A,
            within=pyo.Binary,
            initialize={
                id: lane_separator
                for id, lane_separator in zip(
                    self.network.links_id, self.network.link_lane_separator
                )
            },
        )
        model.non_motor = pyo.Param(
            model.A,
            within=pyo.Binary,
            initialize={
                id: non_motor
                for id, non_motor in zip(
                    self.network.links_id, self.network.link_non_motor
                )
            },
        )
        model.w_p = pyo.Param(
            model.A,
            within=pyo.NonNegativeReals,
            initialize={
                id: non_motor
                for id, non_motor in zip(
                    self.network.links_id, self.network.parking_width
                )
            },
        )
        model.N_ij = pyo.Param(
            model.A,
            within=pyo.NonNegativeIntegers,
            initialize={
                id: N_ij
                for id, N_ij in zip(self.network.links_id, self.network.parking_nums)
            },
        )

        ## OD data
        model.q_rs = pyo.Param(
            model.R,
            model.S,
            within=pyo.NonNegativeReals,
            initialize=lambda model, r, s: self.OD[r - 1][s - 1],
        )

        model.p_curb = 0.05

        ## 2.2 Parking queuing model parameters
        ### Arrival rate:     n_sj/lambda, The number of parking demand on link a
        def init_n_ij(model, i, j):
            total_length = sum(model.l_ij[link] for link in model.A if link[0] == i)
            ij_proportion = model.l_ij[i, j] / total_length if total_length > 0 else 0
            n_ij = (
                model.p_curb
                * sum(model.q_rs[r, s] for (r, s) in model.RS if s == i)
                * ij_proportion
            )
            return int(n_ij)

        model._lambda_ij = pyo.Param(
            model.A, within=pyo.NonNegativeReals, default=0, initialize=init_n_ij
        )

        ### Departure rate:  mu, the average parking duration
        model._mu_ij = 2

        ### Parking rejection probability
        def init_p_ij_N(model, i, j):
            A = model._lambda_ij[i, j] / model._mu_ij
            inverse_p = 1.0
            for m in range(1, model.N_ij[i, j] + 1):
                inverse_p += 1.0 + m / A * inverse_p
            return 1.0 / inverse_p

        model.p_ij_N = pyo.Param(
            model.A,
            within=pyo.NonNegativeReals,
            default=1,
            initialize=init_p_ij_N,
        )

        # 2.3 Parking Adjustment factors
        ### psi: if a lane has lane separator, psi = 0, else psi = 1
        model._psi = pyo.Param(
            model.A,
            within=pyo.Binary,
            default=1,
            initialize={
                idx: 1 - link.lane_separator for idx, link in self.links.items()
            },
        )

        ## delta: if a lane has non-motor vehicle, delta = 0, else delta = 1
        model._delta = pyo.Param(
            model.A,
            within=pyo.Binary,
            default=1,
            initialize={idx: 1 - link.non_motor for idx, link in self.links.items()},
        )

        ## f_l: lane reduction factor
        def init_f_l(model, i, j):
            return (
                1
                - model.w_p[i, j]
                / 9.144
                / model.lane_num[i, j]
                * model._psi[i, j]
                * model._delta[i, j]
            )

        model.f_l = pyo.Param(model.A, within=pyo.NonNegativeReals, initialize=init_f_l)

        ## f_p: parking reduction factor
        def init_f_p(model, i, j):
            return 1 - 0.0012 * model.w_p[i, j] * model._lambda_ij[i, j] * (
                1 - model.p_ij_N[i, j]
            )

        model.f_p = pyo.Param(model.A, within=pyo.NonNegativeReals, initialize=init_f_p)

        # 2.4 Parking demand on each link
        def init_o_ij(model, i, j):
            return model._lambda_ij[i, j] * (1 - model.p_ij_N[i, j]) / model._mu_ij

        model.o_ij = pyo.Expression(model.A, rule=init_o_ij)

        # 2.5 big-M
        def init_big_M(model, i, j):
            # return sum(model.q_rs[r, s] for (r, s) in model.RS)
            return 2 * model.c_0[i, j]

        model.big_M = pyo.Param(
            model.A, within=pyo.NonNegativeReals, initialize=init_big_M
        )

        # 2.6 b_k_r, the node flow conservation parameter
        def init_b_k_r(model, r, k):
            if k == r:
                return sum(model.q_rs[r, s] for s in model.S if s != r)
            else:
                return -model.q_rs[r, k]

        model.b_k_r = pyo.Param(
            model.R, model.S, within=pyo.Reals, initialize=init_b_k_r
        )

        ## capacity under parking reduction factor
        def init_c_1(model, i, j):
            return model.c_0[i, j] * model.f_l[i, j] * model.f_p[i, j]

        model.c_1 = pyo.Param(model.A, within=pyo.NonNegativeReals, initialize=init_c_1)

        # 2.7 Objective weight parameters
        model.w_1 = self.w_1
        model.w_2 = self.w_2
        model.w_3 = self.w_3

        # 3. Variables
        ## 3.1 Parking policy Decision variable, shape: |A|
        model.y_ij = pyo.Var(model.A, domain=pyo.Binary)

        ## 3.2 Link flow variable x_ij, shape: |A|
        def x_ij_0_bounds(model, i, j):
            return (0, 2 * model.c_0[i, j])

        def x_ij_1_bounds(model, i, j):
            return (0, 2 * model.c_1[i, j])

        model.x_ij_0 = pyo.Var(
            model.A, domain=pyo.NonNegativeReals, bounds=x_ij_0_bounds
        )
        model.x_ij_1 = pyo.Var(
            model.A, domain=pyo.NonNegativeReals, bounds=x_ij_1_bounds
        )

        def init_x_ij(model, i, j):
            return model.x_ij_0[i, j] + model.x_ij_1[i, j]

        model.x_ij = pyo.Expression(model.A, rule=init_x_ij)

        # Fix parking policy == allow, for links with lane_separator
        for i, j in model.A:
            if model._psi[i, j] == 0:
                model.y_ij[i, j].fix(1)

        ## 3.3 u_ij and z_ij for linearizing |o_ij - N_ij * y_ij|
        model.u_ij = pyo.Var(model.A, domain=pyo.NonNegativeReals)
        model.z_ij = pyo.Var(model.A, domain=pyo.NonNegativeReals)

        ## 3.4 path-flow variables x_ij_r, shape: |A|*|R|*|S|
        model.x_ij_r = pyo.Var(model.A, model.R, domain=pyo.NonNegativeReals)

        # 4. Constraints
        ## 4.1 u_ij and z_ij constraints
        def unsatisfied_parking_demand_rule(model, i, j):
            return (
                model.u_ij[i, j]
                >= model.o_ij[i, j] - model.N_ij[i, j] * model.y_ij[i, j]
            )

        model.u_ij_constr = pyo.Constraint(
            model.A, rule=unsatisfied_parking_demand_rule
        )

        def unused_parking_lots_rule(model, i, j):
            return (
                model.z_ij[i, j]
                >= model.N_ij[i, j] * model.y_ij[i, j] - model.o_ij[i, j]
            )

        model.z_ij_constr = pyo.Constraint(model.A, rule=unused_parking_lots_rule)

        ## 4.2 Constraints link x_ij and x_ij_0, x_ij_1
        @model.Constraint(model.A)
        def x_ij_0_rule(model, i, j):
            return model.x_ij_0[i, j] <= model.big_M[i, j] * (1 - model.y_ij[i, j])

        @model.Constraint(model.A)
        def x_ij_1_rule(model, i, j):
            return model.x_ij_1[i, j] <= model.big_M[i, j] * model.y_ij[i, j]

        ## 4.3 Node flow conservation constraints
        @model.Constraint(model.R, model.S)
        def node_flow_conservation_rule(model, r, k):
            inflow = sum(model.x_ij_r[i, j, r] for (i, j) in model.A if j == k)
            outflow = sum(model.x_ij_r[i, j, r] for (i, j) in model.A if i == k)
            return outflow - inflow == model.b_k_r[r, k]

        @model.Constraint(model.A)
        def x_ij_r_flow_aggregation_rule(model, i, j):
            return (
                sum(model.x_ij_r[i, j, r] for r in model.R)
                == model.x_ij_0[i, j] + model.x_ij_1[i, j]
            )

        return model

    def solve_UE_TAP(self, epsilon, maxIter):
        """
        Subproblem
        Solve the UE-TAP using Frank-Wolfe algorithm.
        INPUT:
            self.network: Network object
            self.y: fixed  integer variables y_k at iteration k
            self.OD: OD demand matrix
        OUTPUT:
            self.x^*: link flow vector at iteration k
        """

        # reset link flow to zero before each UE-TAP
        self.network.link_time_cost = self.network.link_fft

        TSTT = self.ue_ta(epsilon, maxIter)
        return

    def add_no_good_cut(self):
        model = self.minlp.model
        B_k, N_k = set(), set()

        for i, j in model.A:
            if pyo.value(model.y_ij[i, j]) > 0.5:
                B_k.add((i, j))
            else:
                N_k.add((i, j))
        # self.minlp.model.integer_cut = pyo.Constraint(
        model.no_good_cuts.add(
            expr=sum(model.y_ij[i, j] for (i, j) in model.A if (i, j) in B_k)
            - sum(model.y_ij[i, j] for (i, j) in model.A if (i, j) in N_k)
            <= len(B_k) - 1
        )
        logging.info(f"Add no-good cut: sum(y in B_k) - sum(y in N_k) <= {len(B_k)-1}")

    def add_UE_reduction_cut(self):
        model = self.minlp.model
        rhs = sum(
            (1 - model.y_ij[i, j])
            * model.fft[i, j]
            * (
                self.network.link_flow[idx]
                + 0.03 * self.network.link_flow[idx] ** 5 / model.c_0[i, j] ** 4
            )
            for idx, (i, j) in enumerate(self.network.links_id)
        ) + sum(
            model.y_ij[i, j]
            * model.fft[i, j]
            * (
                self.network.link_flow[idx]
                + 0.03 * self.network.link_flow[idx] ** 5 / model.c_1[i, j] ** 4
            )
            for idx, (i, j) in enumerate(self.network.links_id)
        )
        model.ue_reduction_cuts.add(
            expr=sum(model._kappa[i, j] for i, j in model.A) <= rhs
        )
        logging.info(
            f"Add UE-reduction cut: sum(kappa_ij) <= sum of current UE-TAP objective value"
        )

    def ue_ta(self, epsilon, maxIter):
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


class MINLP:
    def __init__(self):
        self.model = None
        self.epsilon = 5e-3
        self.true_obj_cost = float("inf")
        self.linear_obj_cost = -float("inf")
        self.true_ue_cost = float("inf")
        self.linear_ue_cost = -float("inf")
        self.obj_gap = float("inf")
        self.ue_gap = float("inf")
        self.solver = SolverFactory("gurobi")
        self.solver.options = {
            "MIPGap": self.epsilon,
            "FeasibilityTol": 1e-6,
        }

    def solve_minlp(self):
        self.obj_gap = float("inf")
        self.ue_gap = float("inf")
        i = 1
        while self.obj_gap >= self.epsilon or self.ue_gap >= self.epsilon:
            logging.info(f"--------------MINLP iteration {i} starts--------------")
            # 2. solve the MILP, w.r.t  y_ij, x_ij_0, x_ij_1
            self.add_milp_cut()
            self.solve_milp()
            self.update_gap()
            logging.info(
                f"MINLP Iteration {i} ends, objective linearization gap = {self.obj_gap}, UE-reduction constraint linearization gap= {self.ue_gap}"
            )
            if i == 1:
                self.get_threshold_flow()
            i += 1
        logging.info(
            f"********MINLP solved in {i - 1} iterations, final objective linearization gap= {self.obj_gap}, "
            f"final UE-reduction constraint linearization gap= {self.ue_gap}"
        )
        return

    def build_milp(self):
        model = self.model

        # 0. Initialize x_0_k and x_1_k, the linearization point
        model.x_0_k = pyo.Param(
            model.A, within=pyo.NonNegativeReals, initialize=0, mutable=True
        )
        model.x_1_k = pyo.Param(
            model.A, within=pyo.NonNegativeReals, initialize=0, mutable=True
        )

        # 1. Add auxilary varibles for linearizing objective
        # zeta: objective variable in the MILP master problem
        model._zeta = pyo.Var(model.A, domain=pyo.NonNegativeReals)
        # eta: variable for linearizing UE-reduction cut
        model._kappa = pyo.Var(model.A, domain=pyo.NonNegativeReals, initialize=0.0)

        # zeta >= free flow system optimal cost + u_ij + z_ij
        @model.Constraint(model.A)
        def zeta_minimum_rule(model, i, j):
            return model._zeta[i, j] >= model.fft[i, j] * (
                model.x_ij_0[i, j] + model.x_ij_1[i, j]
            )

        # 2. Add first-order taylor approximation constraints of SO cost
        def x_ij_0_gradient(model, i, j):
            return model.fft[i, j] * (
                1 + 0.75 / model.c_0[i, j] ** 4 * model.x_0_k[i, j] ** 4
            )

        def x_ij_1_gradient(model, i, j):
            return model.fft[i, j] * (
                1 + 0.75 / (model.c_1[i, j]) ** 4 * model.x_1_k[i, j] ** 4
            )

        model.grad_0 = pyo.Param(model.A, initialize=x_ij_0_gradient, mutable=True)
        model.grad_1 = pyo.Param(model.A, initialize=x_ij_1_gradient, mutable=True)

        # 3. Set milp objective
        @model.Objective(sense=pyo.minimize)
        def objective(model):
            return sum(
                model.w_1 * model.u_ij[i, j]
                + model.w_2 * model.z_ij[i, j]
                + model.w_3 * model._zeta[i, j]
                for i, j in model.A
            )

        model.milp_cuts = pyo.ConstraintList()
        model.ue_reduction_cuts = pyo.ConstraintList()
        model.no_good_cuts = pyo.ConstraintList()
        # Initialize ue_reduction cuts:
        for i, j in model.A:
            model.ue_reduction_cuts.add(
                expr=model._kappa[i, j]
                >= model.fft[i, j] * (model.x_ij_0[i, j] + model.x_ij_1[i, j])
            )

        model.x_obj_threshold = pyo.Param(model.A, initialize=0, mutable=True)
        model.x_ue_threshold = pyo.Param(model.A, initialize=0, mutable=True)

    def solve_milp(self):
        model = self.model
        self.result = self.solver.solve(model, tee=False)
        if (
            self.result.solver.status == SolverStatus.ok
            and self.result.solver.termination_condition == TerminationCondition.optimal
        ):
            for i, j in model.A:
                model.x_0_k[i, j].value = max(0, model.x_ij_0[i, j].value)
                model.x_1_k[i, j].value = max(0, model.x_ij_1[i, j].value)
                model.grad_0[i, j].value = model.fft[i, j] * (
                    1 + 0.75 / model.c_0[i, j] ** 4 * model.x_0_k[i, j].value ** 4
                )
                model.grad_1[i, j].value = model.fft[i, j] * (
                    1 + 0.75 / (model.c_1[i, j]) ** 4 * model.x_1_k[i, j].value ** 4
                )
                model.capacity[i, j].value = model.c_0[i, j] * (
                    1 - round(model.y_ij[i, j].value)
                ) + model.c_1[i, j] * round(model.y_ij[i, j].value)
        else:
            logging.info(self.result.solver.termination_condition)

        return

    def add_milp_cut(self):
        model = self.model
        for i, j in model.A:
            # 1. Linearize the SO cost(objective function) at (x_0_k, x_1_k)
            x0k = pyo.value(model.x_0_k[i, j])
            x1k = pyo.value(model.x_1_k[i, j])
            if (
                x0k + x1k > model.x_obj_threshold[i, j].value
                and self.obj_gap > self.epsilon
            ):
                grad_0 = pyo.value(model.grad_0[i, j])
                grad_1 = pyo.value(model.grad_1[i, j])
                rhs = (
                    x0k
                    * model.fft[i, j]
                    * (1 + model._alpha * (x0k / model.c_0[i, j]) ** model._beta)
                    + x1k
                    * model.fft[i, j]
                    * (1 + model._alpha * (x1k / model.c_1[i, j]) ** model._beta)
                    + grad_0 * (model.x_ij_0[i, j] - x0k)
                    + grad_1 * (model.x_ij_1[i, j] - x1k)
                )
                self.model.milp_cuts.add(expr=model._zeta[i, j] >= rhs)

            # 2. Linearize the UE-reduction cut at (x_0_k, x_1_k)
            if (
                x0k + x1k > model.x_ue_threshold[i, j].value
                and self.ue_gap > self.epsilon
            ):
                rhs_ue_cut = (
                    model.fft[i, j] * (x0k + 0.03 * x0k**5 / model.c_0[i, j] ** 4)
                    + model.fft[i, j]
                    * (1 + model._alpha * (x0k / model.c_0[i, j]) ** model._beta)
                    * (model.x_ij_0[i, j] - x0k)
                    + model.fft[i, j] * (x1k + 0.03 * x1k**5 / model.c_1[i, j] ** 4)
                    + model.fft[i, j]
                    * (1 + model._alpha * (x1k / model.c_1[i, j]) ** model._beta)
                    * (model.x_ij_1[i, j] - x1k)
                )
                model.ue_reduction_cuts.add(expr=model._kappa[i, j] >= rhs_ue_cut)

    def update_gap(self):
        model = self.model
        # 1. Update the gap between true objective and linearized objective
        self.true_obj_cost = sum(
            model.x_ij[i, j]()
            * model.fft[i, j]
            * (
                1
                + model._alpha
                * (model.x_ij[i, j]() / model.capacity[i, j].value) ** model._beta
            )
            for i, j in model.A
        )
        # self.true_obj_cost = sum(
        #     model.x_ij_0[i, j].value
        #     * model.fft[i, j]
        #     * (
        #         1
        #         + model._alpha
        #         * (model.x_ij_0[i, j].value / model.c_0[i, j]) ** model._beta
        #     )
        #     + model.x_ij_1[i, j].value
        #     * model.fft[i, j]
        #     * (
        #         1
        #         + model._alpha
        #         * (model.x_ij_1[i, j].value / model.c_1[i, j]) ** model._beta
        #     )
        #     for i, j in model.A
        # )

        self.linear_obj_cost = pyo.value(sum(model._zeta[i, j] for i, j in model.A))
        self.obj_gap = (self.true_obj_cost - self.linear_obj_cost) / self.true_obj_cost
        logging.info(
            "Linearizing SO cost: Current gap: {}, Current true cost: {}, linear cost: {}, ".format(
                round(self.obj_gap,5), self.true_obj_cost, self.linear_obj_cost
            )
        )
        # 2. Update the gap between true UE cost and linearized UE cost
        self.true_ue_cost = sum(
            model.fft[i, j]
            * (
                model.x_ij[i, j]()
                + 0.03 * model.x_ij[i, j]() ** 5 / model.capacity[i, j].value ** 4
            )
            for i, j in model.A
        )
        self.linear_ue_cost = sum(model._kappa[i, j].value for i, j in model.A)
        self.ue_gap = (self.true_ue_cost - self.linear_ue_cost) / self.true_ue_cost

        logging.info(
            "Linearizing UE-reduction constraint: Current gap: {}, true cost: {}, linear cost: {}".format(
                self.ue_gap, self.true_ue_cost, self.linear_ue_cost
            )
        )
        return

    def get_threshold_flow(self):
        """
        Calculate the threshold flow for adding UE-reduction cut, cut won't be added to links whose flow are below this threshold.
        return x_threshold
        """
        model = self.model
        so_cost_threshold = pyo.value(model.objective) * self.epsilon / 2 / len(model.A)
        ue_cost_threshold = (
            sum(model._kappa[i, j].value for i, j in model.A)
            * self.epsilon
            / 2
            / len(model.A)
        )

        def solve_equation(i, j):
            def SO_cost(x):
                return (
                    x
                    * model.fft[i, j]
                    * (
                        1
                        + model._alpha * (x / model.capacity[i, j].value) ** model._beta
                    )
                    - so_cost_threshold
                )

            def UE_cost(x):
                return (
                    model.fft[i, j]
                    * (x + 0.03 * x**5 / model.capacity[i, j].value ** 4)
                    - ue_cost_threshold
                )

            x_obj_threshold = opt.fsolve(SO_cost, 1)[0]
            x_ue_threshold = opt.fsolve(UE_cost, 1)[0]
            return x_obj_threshold, x_ue_threshold

        for i, j in model.A:
            model.x_obj_threshold[i, j].value, model.x_ue_threshold[i, j].value = (
                solve_equation(i, j)
            )

        return

