from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GraphConfig:
    num_cols: int = 4
    nodes_per_col: int = 3
    density: float = 0.7
    cap_min: int = 1
    cap_max: int = 10
    max_line_graph_nodes: int = 100
    num_train_graphs: int = 500

@dataclass
class PolicyConfig:
    embed_dim: int = 128
    network: str = "GAT"  # GCN, GAT, GraphSAGE
    num_layers: int = 2
    pooling_type: str = "mean"
    
@dataclass
class TrainingConfig:
    timesteps: int = 50000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_steps: int = 1024
    batch_size: int = 64
    eval_freq: int = 2048
    n_eval_episodes: int = 20
    seed: int = 42

@dataclass
class ExperimentConfig:
    train_graph_cfg: GraphConfig = field(default_factory=GraphConfig)
    eval_sizes: Tuple[int, ...] = (8, 12, 16, 20) # Total original nodes approximately
    policy_cfg: PolicyConfig = field(default_factory=PolicyConfig)
    train_cfg: TrainingConfig = field(default_factory=TrainingConfig)
