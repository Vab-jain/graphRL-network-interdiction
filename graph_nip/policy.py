from gymnasium import spaces
import torch as th
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable, Any

from graph_nip.gnns import get_network_class
from graph_nip.utils import matrix_features_to_batch

class MatrixObservationToGraph(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        features_dim = 1  # unused
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations) -> Batch:
        node_features = observations["node_features"]
        adj_matrix = observations["adjacency_matrix"]
        return matrix_features_to_batch(node_features, None, adj_matrix)


class GraphActorCriticProcessor(nn.Module):
    def __init__(self, node_dim: int, embed_dim: int = 64, pooling_type: str = "mean", network_kwargs: dict = None, **kwargs):
        super().__init__()
        self.latent_dim_vf = embed_dim
        self.latent_dim_pi = 0  # unused directly by SB3
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type

        processor_class = get_network_class(network_kwargs["network"])
        self.processor = processor_class(
            in_dim=node_dim,
            embed_dim=embed_dim,
            **network_kwargs,
        )

    def _process_graph(self, batch: Batch) -> tuple[th.Tensor, th.Tensor]:
        node_embedding = self.processor(
            node_fts=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        if self.pooling_type == "max":
            graph_embedding = global_max_pool(node_embedding, batch.batch)
        elif self.pooling_type == "mean":
            graph_embedding = global_mean_pool(node_embedding, batch.batch)
        elif self.pooling_type == "sum":
            graph_embedding = global_add_pool(node_embedding, batch.batch)
        return node_embedding, graph_embedding

    def forward(self, batch: Batch) -> tuple[Batch, th.Tensor]:
        node_embedding, graph_embedding = self._process_graph(batch)
        processed_batch = Batch(
            x=node_embedding,
            edge_index=batch.edge_index,
            graph_attr=graph_embedding,
            batch=batch.batch,
        )
        return processed_batch, graph_embedding

    def forward_critic(self, x: Batch) -> th.Tensor:
        return self.forward(x)[1]

    def forward_actor(self, x: Batch) -> Batch:
        return self.forward(x)[0]


class MLPActionHead(nn.Module):
    """
    Action network that predicts per-node logits using an MLP layer.
    Replaces the ProtoActionNetwork from the reference.
    """
    def __init__(self, embed_dim: int, max_nodes: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        
        # We concatenate graph_embedding and node_embedding, so input dim is 2 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1) # Output a single logit scalar per node
        )

    def forward(self, batch: Batch) -> th.Tensor:
        """
        Computes the action logits for each node in the line graph.
        
        The GNN produces representations for both individual nodes and the graph as a whole.
        We combine them so the policy knows both the local node context and the global graph context.

        Args:
            batch (Batch): PyG Batch object containing processed embeddings.
                           - batch.x: Node embeddings.
                           - batch.graph_attr: Graph embeddings.
                           - batch.batch: Batch index mapping nodes to graphs.

        Returns:
            th.Tensor: Dense tensor of shape (batch_size, max_nodes) containing the action logits.
        """
        node_embedding = batch.x
        graph_embedding = batch.graph_attr
        
        # Expand graph_embedding to match number of nodes
        expanded_graph_embedding = graph_embedding[batch.batch]
        
        # Concatenate node and graph embeddings
        combined_features = th.cat([node_embedding, expanded_graph_embedding], dim=1)
        
        # Pass through MLP to get scalar logit per node
        logits = self.mlp(combined_features).squeeze(-1)
        
        # Convert flat tensor to dense batch with padding
        dense_logits, _ = to_dense_batch(logits, batch.batch, fill_value=-1e9, max_num_nodes=self.max_nodes)
        
        return dense_logits


class MaskableGraphActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Callable[[float], float],
        node_dim: int,
        embed_dim: int = 128,
        pooling_type: str = "mean",
        network_kwargs: dict = None,
        *args,
        **kwargs,
    ):
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        self.network_kwargs = network_kwargs

        kwargs.setdefault("features_extractor_class", MatrixObservationToGraph)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        self.action_net = MLPActionHead(
            embed_dim=self.embed_dim,
            max_nodes=self.action_space.n,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GraphActorCriticProcessor(
            node_dim=self.node_dim,
            embed_dim=self.embed_dim,
            pooling_type=self.pooling_type,
            network_kwargs=self.network_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                node_dim=self.node_dim,
                embed_dim=self.embed_dim,
                pooling_type=self.pooling_type,
                network_kwargs=self.network_kwargs,
            )
        )
        return data
