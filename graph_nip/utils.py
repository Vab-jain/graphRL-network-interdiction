import torch as th
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
import inspect
import warnings
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

def flow_network_to_line_graph_data(
    G: nx.DiGraph, 
    current_flow_dict: dict, 
    interdicted_edges: set, 
    remaining_budget_fraction: float
) -> Data:
    """
    Converts a directed flow network into an undirected line graph representation for PyTorch Geometric.
    
    To ensure robust message passing (especially in sparse graphs), we make the line graph *undirected*.
    
    Args:
        G (nx.DiGraph): The original directed flow network.
        current_flow_dict (dict): A nested dictionary of flow values returned by networkx (e.g. flow_dict[u][v]).
        interdicted_edges (set): A set of tuples (u, v) representing edges whose capacities have been set to 0.
        remaining_budget_fraction (float): Float between 0 and 1 indicating how much budget is left.

    Returns:
        torch_geometric.data.Data: PyG Data object containing:
            - x (Tensor): Node features of shape [num_edges_in_G, 4].
            - edge_index (Tensor): Graph connectivity of shape [2, num_line_graph_edges].
    """
    x = []
    # Use frozenset so both (u, v) and (v, u) map to the same edge ID
    id = {frozenset(edge): i for i, edge in enumerate(G.edges())}

    for edge in G.edges():
        cap = G[edge[0]][edge[1]]['capacity']
        
        # Check both directed representations since the environment might store it either way
        is_interdicted = 1.0 if (edge in interdicted_edges or (edge[1], edge[0]) in interdicted_edges) else 0.0
        
        # In undirected graphs, max_flow returns flow in both directions in the dict
        flow_uv = current_flow_dict.get(edge[0], {}).get(edge[1], 0)
        flow_vu = current_flow_dict.get(edge[1], {}).get(edge[0], 0)
        flow = max(flow_uv, flow_vu)
        
        x.append([cap, is_interdicted, flow, remaining_budget_fraction])

    edge_index = []
    for node in G.nodes:
        for edge in G.edges(node):
            for edge_2 in G.edges(node):
                if edge != edge_2:
                    edge_index.append([id[frozenset(edge)], id[frozenset(edge_2)]])
    
    # PyG expects Tensors, and edge_index must be shape [2, num_edges]
    x_tensor = th.tensor(x, dtype=th.float32)
    if edge_index:
        edge_index_tensor = th.tensor(edge_index, dtype=th.long).t().contiguous()
    else:
        edge_index_tensor = th.empty((2, 0), dtype=th.long)
        
    return Data(x=x_tensor, edge_index=edge_index_tensor)


def matrix_features_to_batch(
    node_features: th.Tensor,
    edge_features: th.Tensor | None,
    adj_matrix: th.Tensor,
) -> Batch:
    """
    Convert dense matrix features to a PyTorch Geometric Batch object.
    (Reused from the reference implementation)
    """
    data_list = []
    for b in range(node_features.size(0)):
        edge_index = th.nonzero(adj_matrix[b], as_tuple=False).t()
        edge_attr = (
            edge_features[b][edge_index[0], edge_index[1]]
            if edge_features is not None
            else None
        )
        has_edge = (adj_matrix[b].sum(dim=0) > 0) | (adj_matrix[b].sum(dim=1) > 0)
        
        # Keep nodes that are connected. Isolated nodes are dropped. 
        # For line graphs, all edges usually have connections, but we must handle edge cases.
        node_features_b = node_features[b][has_edge]
        
        # Remap edge_index since we filtered nodes
        node_indices = th.nonzero(has_edge).squeeze(1)
        mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        new_edge_index = th.zeros_like(edge_index)
        for i in range(edge_index.shape[1]):
            new_edge_index[0, i] = mapping[edge_index[0, i].item()]
            new_edge_index[1, i] = mapping[edge_index[1, i].item()]

        data = Data(
            x=node_features_b,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


def get_clean_kwargs(function, warn: bool, kwargs: dict) -> dict:
    """Helper to filter kwargs for a specific function."""
    sampler_args = inspect.signature(function).parameters
    clean_kwargs = {k: kwargs[k] for k in kwargs if k in sampler_args}

    if warn and set(clean_kwargs) != set(kwargs):
        warnings.warn(
            f"Ignoring kwargs {set(kwargs).difference(clean_kwargs)} when calling function {function}",
            UserWarning,
        )

    return clean_kwargs


def change_obs_action_space(
    policy,
    eval_env: VecEnv,
):
    """Update policy observation/action space to match evaluation environment."""
    constructor_args = policy._get_constructor_parameters()
    constructor_args["observation_space"] = eval_env.observation_space
    constructor_args["action_space"] = eval_env.action_space
    eval_policy = policy.__class__(**constructor_args)
    eval_policy.load_state_dict(policy.state_dict())
    return eval_policy
