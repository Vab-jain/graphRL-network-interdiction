# from torch_geometric.typing import weight
import networkx as nx
import numpy as np

def generate_grid_flow_network(
    num_cols: int, 
    nodes_per_col: int, 
    density: float, 
    cap_min: int, 
    cap_max: int, 
    seed: int = None
) -> nx.DiGraph:
    """
    Generates a directed grid-based flow network for the Network Interdiction problem.
    
    The graph is constructed as a grid with a single 'source' node connected to the first column
    and a single 'sink' node connected to the last column. Intermediate columns are populated
    probabilistically based on the density parameter.
    
    Args:
        num_cols (int): Number of intermediate columns in the grid.
        nodes_per_col (int): Maximum number of nodes in each column.
        density (float): Probability of keeping a node in a column (0.0 to 1.0).
        cap_min (int): Minimum capacity for edges.
        cap_max (int): Maximum capacity for edges.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        nx.DiGraph: A directed graph with a 'source' and 'sink' node and 'capacity' attributes on edges.
    """
    if seed:
        np.random.seed(seed)

    if num_cols < 2:
        raise ValueError("Number of intermediate columns must be at least 2")
    
    G = nx.Graph()
    G.add_node('S')
    G.add_node('T')

    node_list = []

    for col in range(num_cols):
        col_node_list = []
        for col_node in range(nodes_per_col):
            if np.random.rand() < density:
                col_node_list.append(f'{col}_{col_node}')
        
        # Ensure at least 1 node per column to maintain connectivity
        if not col_node_list:
            col_node_list.append(f'{col}_0')
            
        node_list.append(col_node_list)
    
    for node in node_list[0]:
        G.add_edge('S', node, capacity=np.random.randint(cap_min, cap_max + 1))
    
    for node in node_list[-1]:
        G.add_edge(node, 'T', capacity=np.random.randint(cap_min, cap_max + 1))
    
    for col in range(num_cols-1):
        for node in node_list[col]:
            for col_node in node_list[col+1]:
                G.add_edge(node, col_node, capacity=np.random.randint(cap_min, cap_max + 1))
    
    return G
