import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any
from torch_geometric.utils import to_dense_adj
import torch as th

from graph_nip.config import GraphConfig
from graph_nip.graph_gen import generate_grid_flow_network
from graph_nip.utils import flow_network_to_line_graph_data

class NetworkInterdictionEnv(gym.Env):
    """
    Gymnasium environment for the Network Interdiction Problem.
    """
    def __init__(self, config: GraphConfig, seed: int = 42, is_eval: bool = False):
        super().__init__()
        self.config = config
        self.np_random = np.random.RandomState(seed)
        self.is_eval = is_eval
        
        self.max_line_graph_nodes = self.config.max_line_graph_nodes
        
        # Action space: index of the line graph node (which represents an edge in the original graph)
        self.action_space = gym.spaces.Discrete(self.max_line_graph_nodes)
        
        # Observation space: node features and adjacency matrix of the line graph
        # Node features: 4 dimensions [capacity, is_interdicted, current_flow, remaining_budget_fraction]
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                low=0.0, high=float('inf'), 
                shape=(self.max_line_graph_nodes, 4), dtype=np.float32
            ),
            "adjacency_matrix": gym.spaces.Box(
                low=0, high=1, 
                shape=(self.max_line_graph_nodes, self.max_line_graph_nodes), dtype=np.float32
            )
        })
        from collections import OrderedDict
        self.observation_space.spaces = OrderedDict(self.observation_space.spaces)
        
        self.graphs = [self._generate_graph() for _ in range(config.num_train_graphs if not is_eval else 20)]
        self.current_graph_idx = 0
        
    def _generate_graph(self) -> nx.DiGraph:
        """Generates a base directed flow network."""
        return generate_grid_flow_network(
            num_cols=self.config.num_cols,
            nodes_per_col=self.config.nodes_per_col,
            density=self.config.density,
            cap_min=self.config.cap_min,
            cap_max=self.config.cap_max,
            seed=self.np_random.randint(0, 100000)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Select the next graph
        self.G = self.graphs[self.current_graph_idx].copy()
        self.current_graph_idx = (self.current_graph_idx + 1) % len(self.graphs)
        
        # Find source and sink nodes (assumed to be node 0 and the max integer node, or by attributes if added)
        # Assuming generate_grid_flow_network produces nodes as integers 0 to N.
        # Let's define them explicitly to be safe: min node is source, max is sink.
        nodes = list(self.G.nodes())
        self.source = min(nodes)
        self.sink = max(nodes)
        
        # Initial max flow calculation
        self.initial_max_flow, self.current_flow_dict = nx.maximum_flow(
            self.G, _s=self.source, _t=self.sink, capacity='capacity'
        )
        self.current_max_flow = self.initial_max_flow
        
        # State tracking
        self.interdicted_edges = set()
        self.num_edges = self.G.number_of_edges()
        
        # Budget = max(1, num_edges // 4)
        self.max_steps = max(1, self.num_edges // 4)
        self.elapsed_steps = 0
        
        # We need a stable mapping from line graph node indices to original edges
        self.edge_list = list(self.G.edges())
        
        return self._get_observation(), {}
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Generate line graph PyG Data object
        remaining_budget = (self.max_steps - self.elapsed_steps) / self.max_steps
        data = flow_network_to_line_graph_data(
            self.G, self.current_flow_dict, self.interdicted_edges, remaining_budget
        )
        
        # Convert PyG Data to padded matrices for SB3
        node_features = np.zeros((self.max_line_graph_nodes, 4), dtype=np.float32)
        num_nodes = data.x.size(0)
        node_features[:num_nodes, :] = data.x.numpy()
        
        adj_matrix = np.zeros((self.max_line_graph_nodes, self.max_line_graph_nodes), dtype=np.float32)
        if data.edge_index.numel() > 0:
            dense_adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0).numpy()
            adj_matrix[:num_nodes, :num_nodes] = dense_adj
            
        return {
            "node_features": node_features,
            "adjacency_matrix": adj_matrix
        }

    def _apply_interdiction_and_compute_reward(self, action_edge: Tuple[Any, Any]) -> float:
        """
        Applies the interdiction action to the graph and computes the dense reward.
        
        The reward is formulated as the marginal reduction in flow caused by this specific action.
        Reward = (previous_max_flow - new_max_flow) / initial_max_flow

        Args:
            action_edge (Tuple[Any, Any]): The edge `(u, v)` to interdict.

        Returns:
            float: The reward for this step.
        """
        self.G[action_edge[0]][action_edge[1]]['capacity'] = 0
        self.interdicted_edges.add(action_edge)
        self.interdicted_edges.add((action_edge[1], action_edge[0]))

        new_max_flow_value, new_flow_dict = nx.maximum_flow(self.G, _s=self.source, _t=self.sink, capacity='capacity')

        self.current_flow_dict = new_flow_dict

        r = ((self.current_max_flow - new_max_flow_value)  / self.initial_max_flow) if self.initial_max_flow else 0

        self.current_max_flow = new_max_flow_value

        return float(r)

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean array masking out invalid actions.
        
        An action is valid if:
        1. It points to a valid line graph node (index < self.num_edges).
        2. The edge corresponding to this node has NOT been interdicted yet.
        
        Returns:
            np.ndarray: A boolean array of shape (max_line_graph_nodes,) where True indicates a valid action.
        """
        mask = np.zeros(self.max_line_graph_nodes, dtype=bool)
        for i, edge in enumerate(self.edge_list):
            if edge not in self.interdicted_edges:
                mask[i] = True
        return mask

    def step(self, action):
        if action >= self.num_edges:
            raise ValueError(f"Invalid action {action}, max valid is {self.num_edges-1}")
            
        action_edge = self.edge_list[action]
        
        # Execute action and get reward
        reward = self._apply_interdiction_and_compute_reward(action_edge)
        
        self.elapsed_steps += 1
        terminated = self.elapsed_steps >= self.max_steps
        truncated = False
        
        info = {}
        if terminated:
            info["final_flow"] = self.current_max_flow
            info["flow_reduction_ratio"] = (self.initial_max_flow - self.current_max_flow) / (self.initial_max_flow + 1e-9)
            
        return self._get_observation(), float(reward), terminated, truncated, info
