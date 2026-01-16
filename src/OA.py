import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import logging
import itertools
import scipy.optimize as opt

from src.DNDP import DNDP

class OA(DNDP):
    """
    subclass of DNDP, implement the Outer Approximation algorithm for solving the DNDP problem.
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
        self.RMP = RMIL_MP()
        self.RMP.model = self.build_base_model()
        self.RMP.build_RMIL_MP()

    def solve_model(self):
        LB, UB = 0, float("inf")
        gap = float("inf")

        # 1. Initialize a feasible y_iter on the upper-level
        k = 1
        while gap > self.dndp_epsilon:
            logging.info(
                f"############### Outer Approximation Iteration {k} starts, current gap: {gap}, LB: {LB}, UB: {UB} ###############"
            )
            # 2. Solve the Mixed Integer Linear master problem, obtain y^k and LB^k, update LB.
            self.RMP.solve_RMIL_MP()

            ## IF infeasible, terminate, return best solution found so far.
            temp_LB = self.RMP.model.ObjVal

            if temp_LB < LB * (1 - self.dndp_epsilon):
                logging.warning(
                    "!!!!RMILP_MP solver returns a worse LB, this indicates an error with ue-reduction cut or no-good cut!!!!")
            LB = max(temp_LB, LB)

            if LB >= UB * (1 - self.dndp_epsilon):
                self.LB_record.append(LB)
                self.UB_record.append(self.UB_record[-1])
                logging.info(
                    f"############### Outer Approximation Iteration {k} ends, LB={LB} >= UB*(1-epsilon)={UB * (1 - self.dndp_epsilon)} ############### \n"
                    f"~~~~~~~~~~~~~Outer Approximation converged at iteration: {k}, reach goal gap: {self.dndp_epsilon}~~~~~~~~~~~~~"
                )
                break

            # 3. Solve the UE subproblem with fixed y^k, obtain x^k and UB^k, update UB
            self.network.link_capacity = np.array(
                [self.RMP.capacity[i, j] for i, j in self.RMP.A]
            )
            ue_cost = self.solve_subproblem(epsilon=1e-4, maxIter=1000)

            current_y = {(i, j): self.RMP.y_ij[i, j].X for i, j in self.RMP.A}
            self.RMP.update_linearization_point(self.network.link_flow, current_y)

            temp_UB = (
                    self.w_1
                    * sum(self.RMP.u_ij[i, j].X for i, j in self.RMP.A)
                    + self.w_2
                    * sum(self.RMP.z_ij[i, j].X for i, j in self.RMP.A)
                    + self.w_3 * ue_cost
            )
            if temp_UB < UB:
                UB = temp_UB
                self.best_solution = [
                    self.RMP.y_ij[i, j].X
                    for i, j in self.RMP.A
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
                    f"############### Outer Approximation Iteration {k} ends, LB={LB} >= UB*(1-epsilon)={UB * (1 - self.dndp_epsilon)} ############### \n"
                    f"~~~~~~~~~~~~~Outer Approximation converged at iteration: {k}, reach goal gap: {self.dndp_epsilon}~~~~~~~~~~~~~ \n"
                    f"Best solution saved to best_solution.csv, "
                )
                self.UB_record.append(self.UB_record[-1])
                break

            gap = round(abs(1 - LB / UB), 5)
            logging.info(
                f"############### Outer Approximation Iteration {k} ends, current gap: {gap}, LB: {LB}, UB: {UB} ###############"
            )

            k += 1

        ## Save best solution and LB and UB record
        np.savetxt("best_solution.csv", np.array(self.best_solution), delimiter=",")
        convergence_record = pd.DataFrame(
            {"LB": self.LB_record, "UB": self.UB_record}
        )
        convergence_record.to_csv("convergence_record.csv", index=False)

    def build_base_model(self):
        # Change: Use Gurobi Model
        model = gp.Model("DNDP_Base")
        model.Params.LogToConsole = 0

        # Change: Use self.RMP to store data (Simulating Pyomo model.param storage)
        rmp = self.RMP

        # 1. Construct sets
        ## Node set
        rmp.R = list(self.nodes.keys())
        rmp.S = list(self.nodes.keys())
        rs = list(itertools.product(rmp.R, repeat=2))
        rmp.RS = rs

        ## Link set
        rmp.A = [link for link in self.links.keys()]

        # 2. Construct parameters
        # Change: Using Python Dicts instead of pyo.Param

        # 2.1 Network parameters
        rmp.l_ij = {
            id: length
            for id, length in zip(self.network.links_id, self.network.link_length)
        }

        rmp.l_0 = 100  # length that are not allowed for on-street parking on a road segment near intersections
        rmp.c_0 = {
            id: c_0 for id, c_0 in zip(self.network.links_id, self.network.link_c_0)
        }

        rmp.capacity = {
            id: capacity
            for id, capacity in zip(
                self.network.links_id, self.network.link_capacity
            )
        }

        rmp.fft = {
            id: link_fft
            for id, link_fft in zip(self.network.links_id, self.network.link_fft)
        }

        rmp._alpha = 0.15
        rmp._beta = 4
        rmp.lane_num = {
            id: link_lane_num
            for id, link_lane_num in zip(
                self.network.links_id, self.network.link_lane_num
            )
        }

        rmp.road_width = {
            id: road_width
            for id, road_width in zip(
                self.network.links_id, self.network.link_road_width
            )
        }

        rmp.lane_separator = {
            id: lane_separator
            for id, lane_separator in zip(
                self.network.links_id, self.network.link_lane_separator
            )
        }

        rmp.non_motor = {
            id: non_motor
            for id, non_motor in zip(
                self.network.links_id, self.network.link_non_motor
            )
        }

        rmp.w_b = 4.0
        rmp.w_p = {
            id: non_motor
            for id, non_motor in zip(
                self.network.links_id, self.network.parking_width
            )
        }

        rmp.N_ij = {
            id: N_ij
            for id, N_ij in zip(self.network.links_id, self.network.parking_nums)
        }

        ## OD data
        rmp.q_rs = {(r, s): self.OD[r - 1][s - 1] for r in rmp.R for s in rmp.S}

        rmp.p_curb = 0.05

        ## 2.2 Parking queuing model parameters
        ### Arrival rate:     n_sj/lambda, The number of parking demand on link a
        rmp._lambda_ij = {}
        for i, j in rmp.A:
            total_length = sum(rmp.l_ij[link] for link in rmp.A if link[0] == i)
            ij_proportion = rmp.l_ij[i, j] / total_length if total_length > 0 else 0
            n_ij = (
                    rmp.p_curb
                    * sum(rmp.q_rs[r, s] for (r, s) in rmp.RS if s == i)
                    * ij_proportion
            )
            rmp._lambda_ij[i, j] = int(n_ij)

        ### Departure rate:  mu, the average parking duration
        rmp._mu_ij = 2

        ### Parking rejection probability
        rmp.p_ij_N = {}
        for i, j in rmp.A:
            A = rmp._lambda_ij[i, j] / rmp._mu_ij
            inverse_p = 1.0
            for m in range(1, rmp.N_ij[i, j] + 1):
                inverse_p += 1.0 + m / A * inverse_p
            rmp.p_ij_N[i, j] = 1.0 / inverse_p

        # 2.3 Parking demand on each link
        # Change: Calculated immediately and stored as dict (was Pyomo Expression)
        rmp.L_ij = {}
        for i, j in rmp.A:
            rmp.L_ij[i, j] = rmp._lambda_ij[i, j] * (1 - rmp.p_ij_N[i, j]) / rmp._mu_ij

        # 2.4 Parking Adjustment factors
        ### delta: if a lane has lane separator, psi = 0, else psi = 1
        rmp._delta = {
            idx: link.lane_separator for idx, link in self.links.items()
        }

        ## psi: if a lane has non-motor vehicle, delta = 0, else delta = 1
        rmp._psi = {idx: link.non_motor for idx, link in self.links.items()}

        ## f_l: lane reduction factor
        rmp.f_l = {}
        for i, j in rmp.A:
            rmp.f_l[i, j] = (
                    1
                    + ((1 - rmp._psi[i, j]) * rmp.w_b - rmp.w_p[i, j])
                    / (9.144 * rmp.lane_num[i, j])
                    * (1 - rmp._delta[i, j])
            )

        ## f_p: parking reduction factor
        rmp.f_p = {}
        for i, j in rmp.A:
            rmp.f_p[i, j] = 1 - 0.0012 * 6 * rmp.L_ij[i, j] * (
                        1 - rmp._delta[i, j])  # 6 denotes length of a parking space in meters

        # 2.5 big-M
        rmp.big_M = {}
        for i, j in rmp.A:
            rmp.big_M[i, j] = 2 * rmp.c_0[i, j]

        # 2.6 b_k_r, the node flow conservation parameter
        rmp.b_k_r = {}
        for r in rmp.R:
            for k in rmp.S:
                if k == r:
                    rmp.b_k_r[r, k] = sum(rmp.q_rs[r, s] for s in rmp.S if s != r)
                else:
                    rmp.b_k_r[r, k] = -rmp.q_rs[r, k]

        ## capacity under parking reduction factor
        rmp.c_1 = {}
        for i, j in rmp.A:
            rmp.c_1[i, j] = rmp.c_0[i, j] * rmp.f_l[i, j] * rmp.f_p[i, j]

        # 2.7 Objective weight parameters
        rmp.w_1 = self.w_1
        rmp.w_2 = self.w_2
        rmp.w_3 = self.w_3

        # 3. Variables
        ## 3.1 Parking policy Decision variable, shape: |A|
        rmp.y_ij = model.addVars(rmp.A, vtype=GRB.BINARY, name="y_ij")

        ## 3.2 Link flow variable x_ij, shape: |A|
        rmp.x_ij_0 = model.addVars(rmp.A, lb=0.0, name="x_ij_0")
        rmp.x_ij_1 = model.addVars(rmp.A, lb=0.0, name="x_ij_1")

        # Setting bounds
        for i, j in rmp.A:
            rmp.x_ij_0[i, j].UB = 2 * rmp.c_0[i, j]
            rmp.x_ij_1[i, j].UB = 2 * rmp.c_1[i, j]

        # Simulating Expression x_ij
        rmp.x_ij = {k: rmp.x_ij_0[k] + rmp.x_ij_1[k] for k in rmp.A}

        # Fix parking policy == allow, for links with lane_separator
        for i, j in rmp.A:
            if rmp._delta[i, j] == 0:
                rmp.y_ij[i, j].lb = 1
                rmp.y_ij[i, j].ub = 1

        ## 3.3 u_ij and z_ij for linearizing |L_ij - N_ij * y_ij|
        rmp.u_ij = model.addVars(rmp.A, lb=0.0, name="u_ij")
        rmp.z_ij = model.addVars(rmp.A, lb=0.0, name="z_ij")

        ## 3.4 path-flow variables x_ij_r, shape: |A|*|R|*|S|
        rmp.x_ij_r = model.addVars(rmp.A, rmp.R, lb=0.0, name="x_ij_r")

        # 4. Constraints
        ## 4.1 u_ij and z_ij constraints
        model.addConstrs(
            (rmp.u_ij[i, j] >= rmp.L_ij[i, j] - rmp.N_ij[i, j] * rmp.y_ij[i, j] for i, j in rmp.A),
            name="unsatisfied_parking_demand_rule"
        )

        model.addConstrs(
            (rmp.z_ij[i, j] >= rmp.N_ij[i, j] * rmp.y_ij[i, j] - rmp.L_ij[i, j] for i, j in rmp.A),
            name="unused_parking_lots_rule"
        )

        ## 4.2 Constraints link x_ij and x_ij_0, x_ij_1
        model.addConstrs(
            (rmp.x_ij_0[i, j] <= rmp.big_M[i, j] * (1 - rmp.y_ij[i, j]) for i, j in rmp.A),
            name="x_ij_0_rule"
        )

        model.addConstrs(
            (rmp.x_ij_1[i, j] <= rmp.big_M[i, j] * rmp.y_ij[i, j] for i, j in rmp.A),
            name="x_ij_1_rule"
        )

        ## 4.3 Node flow conservation constraints
        model.addConstrs(
            (gp.quicksum(rmp.x_ij_r[i, j, r] for (i, j) in rmp.A if i == k) -
             gp.quicksum(rmp.x_ij_r[i, j, r] for (i, j) in rmp.A if j == k) == rmp.b_k_r[r, k]
             for r in rmp.R for k in rmp.S),
            name="node_flow_conservation_rule"
        )

        model.addConstrs(
            (gp.quicksum(rmp.x_ij_r[i, j, r] for r in rmp.R) == rmp.x_ij_0[i, j] + rmp.x_ij_1[i, j]
             for i, j in rmp.A),
            name="x_ij_r_flow_aggregation_rule"
        )

        return model

    def solve_subproblem(self, epsilon, maxIter):
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

        ue_cost = self.solve_UE_TAP(epsilon, maxIter)
        logging.info(f"Non-linear Subproblem solved")
        return ue_cost

    def add_no_good_cut(self):
        # Change: accessing vars from self.RMP
        model = self.RMP.model
        rmp = self.RMP
        B_k, N_k = [], []

        for i, j in rmp.A:
            if rmp.y_ij[i, j].X > 0.5:
                B_k.append((i, j))
            else:
                N_k.append((i, j))

        model.addConstr(
            gp.quicksum(rmp.y_ij[i, j] for (i, j) in B_k)
            - gp.quicksum(rmp.y_ij[i, j] for (i, j) in N_k)
            <= len(B_k) - 1
        )
        logging.info(f"Add no-good cut: sum(y in B_k) - sum(y in N_k) <= {len(B_k) - 1}")

    def add_UE_reduction_cut(self):

        model = self.RMP.model
        rmp = self.RMP

        rhs_val = 0
        for idx, (i, j) in enumerate(self.network.links_id):
            # Recalculate expression for RHS since we don't use Pyomo expressions
            flow = self.network.link_flow[idx]
            term_0 = rmp.fft[i, j] * (flow + 0.03 * flow ** 5 / rmp.c_0[i, j] ** 4)
            term_1 = rmp.fft[i, j] * (flow + 0.03 * flow ** 5 / rmp.c_1[i, j] ** 4)
            rhs_val += (1 - rmp.y_ij[i,j]) * term_0 + rmp.y_ij[i,j] * term_1

        model.addConstr(
            gp.quicksum(rmp._kappa[i, j] for i, j in rmp.A) <= rhs_val,
            name=f"ue_reduction_cut_{len(model.getConstrs())}"
        )
        logging.info(
            f"Add UE-reduction cut: sum(kappa_ij) <= sum of current UE-TAP objective value"
        )


class RMIL_MP:
    """
    Relaxed Master Integer Linear Programming (RMIL_MP) problem for solving the DNDP using Outer Approximation algorithm.
    """

    def __init__(self):
        self.model = None
        self.epsilon = 1e-4
        self.true_obj_cost = float("inf")
        self.linear_obj_cost = -float("inf")
        self.true_ue_cost = float("inf")
        self.linear_ue_cost = -float("inf")
        self.obj_gap = float("inf")
        self.ue_gap = float("inf")
        # Change: Solver factory removed, Gurobi model handles itself

    def solve_RMIL_MP(self):
        self.obj_gap = float("inf")
        self.ue_gap = float("inf")
        i = 1

        # Change: Set params on the model object
        self.model.Params.MIPGap = self.epsilon
        self.model.Params.FeasibilityTol = 1e-6

        while self.obj_gap >= self.epsilon or self.ue_gap >= self.epsilon:
            logging.info(
                f"--------------Relaxed Mixed Integer Linear Programming Master Problem iteration {i} starts--------------")
            # 2. solve the MILP, w.r.t  y_ij, x_ij_0, x_ij_1
            self.add_milp_cut()
            self.solve_milp()
            self.update_gap()
            logging.info(
                f"Relaxed Mixed Integer Linear Programming Master Problem Iteration {i} ends, objective linearization gap = {self.obj_gap}, UE-reduction constraint linearization gap= {self.ue_gap}"
            )
            if i == 1:
                self.get_threshold_flow()
            i += 1
        logging.info(
            f"********Relaxed Mixed Integer Linear Programming Master Problem solved in {i - 1} iterations, final objective linearization gap= {self.obj_gap}, "
            f"final UE-reduction constraint linearization gap= {self.ue_gap}"
        )
        return

    def build_RMIL_MP(self):
        model = self.model

        # 0. Initialize x_0_k and x_1_k, the linearization point
        # Change: Python dicts used instead of Pyomo Param(mutable=True)
        self.x_0_k = {k: 0.0 for k in self.A}
        self.x_1_k = {k: 0.0 for k in self.A}

        # 1. Add auxilary varibles for linearizing objective
        # zeta: objective variable in the MILP master problem
        self._zeta = model.addVars(self.A, lb=0.0, name="zeta")
        # eta: variable for linearizing UE-reduction cut
        self._kappa = model.addVars(self.A, lb=0.0, name="kappa")

        # zeta >= free flow system optimal cost + u_ij + z_ij
        model.addConstrs(
            (self._zeta[i, j] >= self.fft[i, j] * (self.x_ij_0[i, j] + self.x_ij_1[i, j])
             for i, j in self.A),
            name="zeta_minimum_rule"
        )

        # 2. Add first-order taylor approximation constraints of SO cost
        # Change: Using simple Python functions/dicts
        def x_ij_0_gradient(i, j):
            return self.fft[i, j] * (
                    1 + 0.75 / self.c_0[i, j] ** 4 * self.x_0_k[i, j] ** 4
            )

        def x_ij_1_gradient(i, j):
            return self.fft[i, j] * (
                    1 + 0.75 / (self.c_1[i, j]) ** 4 * self.x_1_k[i, j] ** 4
            )

        self.grad_0 = {k: x_ij_0_gradient(*k) for k in self.A}
        self.grad_1 = {k: x_ij_1_gradient(*k) for k in self.A}

        # 3. Set milp objective
        # Change: Gurobi setObjective
        obj_expr = gp.quicksum(
            self.w_1 * self.u_ij[i, j]
            + self.w_2 * self.z_ij[i, j]
            + self.w_3 * self._zeta[i, j]
            for i, j in self.A
        )
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Initialize ue_reduction cuts:
        for i, j in self.A:
            model.addConstr(
                self._kappa[i, j] >= self.fft[i, j] * (self.x_ij_0[i, j] + self.x_ij_1[i, j])
            )

        self.x_obj_threshold = {k: 0.0 for k in self.A}
        self.x_ue_threshold = {k: 0.0 for k in self.A}

    def solve_milp(self):
        model = self.model
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            for i, j in self.A:
                self.x_0_k[i, j] = max(0, self.x_ij_0[i, j].X)
                self.x_1_k[i, j] = max(0, self.x_ij_1[i, j].X)
                self.grad_0[i, j] = self.fft[i, j] * (
                        1 + 0.75 / self.c_0[i, j] ** 4 * self.x_0_k[i, j] ** 4
                )
                # Gradient for x_ij_1 (Capacity C_1)
                self.grad_1[i, j] = self.fft[i, j] * (
                        1 + 0.75 / self.c_1[i, j] ** 4 * self.x_1_k[i, j] ** 4
                )

                y_val = round(self.y_ij[i, j].X)
                self.capacity[i, j] = self.c_0[i, j] * (
                        1 - y_val
                ) + self.c_1[i, j] * y_val
        else:
            logging.info(f"Solver Status: {model.Status}")

        return

    def add_milp_cut(self):
        model = self.model
        for i, j in self.A:
            # 1. Linearize the SO cost(objective function) at (x_0_k, x_1_k)
            x0k = self.x_0_k[i, j]
            x1k = self.x_1_k[i, j]
            if (
                    x0k + x1k > self.x_obj_threshold[i, j]
                    and self.obj_gap > self.epsilon
            ):
                grad_0 = self.grad_0[i, j]
                grad_1 = self.grad_1[i, j]
                rhs = (
                        x0k
                        * self.fft[i, j]
                        * (1 + self._alpha * (x0k / self.c_0[i, j]) ** self._beta)
                        + x1k
                        * self.fft[i, j]
                        * (1 + self._alpha * (x1k / self.c_1[i, j]) ** self._beta)
                        + grad_0 * (self.x_ij_0[i, j] - x0k)
                        + grad_1 * (self.x_ij_1[i, j] - x1k)
                )
                model.addConstr(self._zeta[i, j] >= rhs)

            # 2. Linearize the UE-reduction cut at (x_0_k, x_1_k)
            if (
                    x0k + x1k > self.x_ue_threshold[i, j]
                    and self.ue_gap > self.epsilon
            ):
                rhs_ue_cut = (
                        self.fft[i, j] * (x0k + 0.03 * x0k ** 5 / self.c_0[i, j] ** 4)
                        + self.fft[i, j]
                        * (1 + self._alpha * (x0k / self.c_0[i, j]) ** self._beta)
                        * (self.x_ij_0[i, j] - x0k)
                        + self.fft[i, j] * (x1k + 0.03 * x1k ** 5 / self.c_1[i, j] ** 4)
                        + self.fft[i, j]
                        * (1 + self._alpha * (x1k / self.c_1[i, j]) ** self._beta)
                        * (self.x_ij_1[i, j] - x1k)
                )
                model.addConstr(self._kappa[i, j] >= rhs_ue_cut)

    def update_linearization_point(self, link_flows, y_fixed):
        """
        Updates x_0_k and x_1_k using the true feasible flows from the UE Subproblem.
        Crucial for correct Outer Approximation cuts.
        """
        for idx, (i, j) in enumerate(self.A):
            true_flow = link_flows[idx]

            # Use the y from the solution we just evaluated
            # (If y[i,j] is 1, flow belongs to x_1, else x_0)
            if y_fixed[i, j] > 0.5:
                self.x_0_k[i, j] = 0.0
                self.x_1_k[i, j] = true_flow
            else:
                self.x_0_k[i, j] = true_flow
                self.x_1_k[i, j] = 0.0

            # Update gradients immediately for the next cut generation
            self.grad_0[i, j] = self.fft[i, j] * (
                    1 + 0.75 / self.c_0[i, j] ** 4 * self.x_0_k[i, j] ** 4
            )
            self.grad_1[i, j] = self.fft[i, j] * (
                    1 + 0.75 / (self.c_1[i, j]) ** 4 * self.x_1_k[i, j] ** 4
            )

    def update_gap(self):
        model = self.model
        # 1. Update the gap between true objective and linearized objective
        self.true_obj_cost = sum(
            (self.x_ij_0[i, j].X + self.x_ij_1[i, j].X)
            * self.fft[i, j]
            * (
                    1
                    + self._alpha
                    * ((self.x_ij_0[i, j].X + self.x_ij_1[i, j].X) / self.capacity[i, j]) ** self._beta
            )
            for i, j in self.A
        )

        self.linear_obj_cost = sum(self._zeta[i, j].X for i, j in self.A)
        # Avoid division by zero
        if self.true_obj_cost != 0:
            self.obj_gap = (self.true_obj_cost - self.linear_obj_cost) / self.true_obj_cost
        else:
            self.obj_gap = 0

        logging.info(
            "Linearizing SO cost: Current gap: {}, Current true cost: {}, linear cost: {}, ".format(
                round(self.obj_gap, 5), self.true_obj_cost, self.linear_obj_cost
            )
        )
        # 2. Update the gap between true UE cost and linearized UE cost
        self.true_ue_cost = sum(
            self.fft[i, j]
            * (
                    (self.x_ij_0[i, j].X + self.x_ij_1[i, j].X)
                    + 0.03 * (self.x_ij_0[i, j].X + self.x_ij_1[i, j].X) ** 5 / self.capacity[i, j] ** 4
            )
            for i, j in self.A
        )
        self.linear_ue_cost = sum(self._kappa[i, j].X for i, j in self.A)

        if self.true_ue_cost != 0:
            self.ue_gap = (self.true_ue_cost - self.linear_ue_cost) / self.true_ue_cost
        else:
            self.ue_gap = 0

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

        so_cost_threshold = model.ObjVal * self.epsilon / 2 / len(self.A)
        ue_cost_threshold = (
                sum(self._kappa[i, j].X for i, j in self.A)
                * self.epsilon
                / 2
                / len(self.A)
        )

        def solve_equation(i, j):
            # Change: accessing capacity from dict
            c_val = self.capacity[i, j]

            def SO_cost(x):
                return (
                        x
                        * self.fft[i, j]
                        * (
                                1
                                + self._alpha * (x / c_val) ** self._beta
                        )
                        - so_cost_threshold
                )

            def UE_cost(x):
                return (
                        self.fft[i, j]
                        * (x + 0.03 * x ** 5 / c_val ** 4)
                        - ue_cost_threshold
                )

            x_obj_threshold = opt.fsolve(SO_cost, 1)[0]
            x_ue_threshold = opt.fsolve(UE_cost, 1)[0]
            return x_obj_threshold, x_ue_threshold

        for i, j in self.A:
            self.x_obj_threshold[i, j], self.x_ue_threshold[i, j] = (
                solve_equation(i, j)
            )

        return