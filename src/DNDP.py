import numpy as np
import pandas as pd
from src.network import Node, Link, Network

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
        self.minlp = None
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