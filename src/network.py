import numpy as np
from dataclasses import dataclass, field

@dataclass(slots=True)
class Node:
    id: int
    inlinks: list = field(default_factory=list)
    outlinks: list = field(default_factory=list)


class Link:
    def __init__(self, link):
        self.init_node = link["init_node"]
        self.term_node = link["term_node"]
        self.capacity = link["capacity"]
        self.length = link["length"]
        self.fft = link["free_flow_time"]
        self.alpha = link["b"]
        self.beta = link["power"]
        self.flow = 0.0
        self.time_cost = self.fft
        self.lane_num = link["lane_num"]
        self.road_width = link["road_width"]
        self.lane_separator = link["lane_separator"]
        self.non_motor = link["non_motor"]


class Network:
    def __init__(self, nodes, links, link_df):
        self.nodes = nodes
        self.links = links
        self.links_id = list(self.links.keys())
        self.link_c_0 = link_df["capacity"].to_numpy()
        self.link_length = link_df["length"].to_numpy()
        self.link_fft = link_df["free_flow_time"].to_numpy()
        self.link_b = link_df["b"].to_numpy()
        self.link_power = link_df["power"].to_numpy()
        self.link_lane_num = link_df["lane_num"].to_numpy()
        self.link_road_width = link_df["road_width"].to_numpy()
        self.link_lane_separator = link_df["lane_separator"].to_numpy()
        self.link_non_motor = link_df["non_motor"].to_numpy()
        self.parking_nums = link_df["parking_num"].to_numpy()
        self.parking_width = link_df["parking_width"].to_numpy()
        self.link_flow = np.zeros_like(self.link_c_0).astype(float)
        self.link_capacity = self.link_c_0.copy()
        self.link_time_cost = self.link_fft.copy()