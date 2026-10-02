from dataclasses import dataclass 
import torch 
from typing import Self

from ..exceptions import InvalidGraphStructure

@dataclass 
class GraphStructure:
    """ 
    Graph Structure class; the pytorch representation of a graph.

    Parameters
    -----------
    ``edge_index`` : torch.Tensor
        Tensor object depicting which nodes are connected (shape: [2, num_edges])
    ``edge_weight`` : torch.Tensor
        Tensor object depicting the weight of edges (shape: [num_edges])
    ``num_nodes`` : int
        Number of nodes represented in the graph. This should include also isolated
        nodes that would not show up in ``edge_index``.

    Attributes
    -----------
    ``num_edges``
        Number of edges in the graph.
    ``adjacency_matrix``
        Adjacency matrix: nodes in rows and columns, with value of associated 
        edge-weight.
    ``edge_index_list``
        List representation of ``edge_index``.
    ``edge_weight_list``
        List representation of ``edge_weight``.
    ``density``
        Proportion of all possible connections that exist:
        num_edges / (num_nodes * (num_nodes - 1))
    ``mean_degree``
        Average number of connections per node:
        num_edges / num_nodes

    Downstream
    ----------
    While this is ultimately the graph that GNNs work with, these are not worked with
    in unison. ``GraphStructure``s are wrapped in ``GraphObject``s.

    Examples
    ---------
    >>> edges   = [(0,1), (1,0), (3,1)]
    ... graph   = GraphStructure.from_list(edges, [1, 1, 1], 4)
    """
    edge_index:     torch.Tensor
    edge_weight:    torch.Tensor 
    num_nodes:      int

    def __post_init__(self):
        self._validate()

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]
    
    @property
    def adjacency_matrix(self) -> torch.Tensor:
        return self._get_adjacency_matrix()
        
    @property
    def edge_index_list(self) -> list[tuple[int, int]]:
        return [tuple(edge) for edge in self.edge_index.tolist()]

    @property
    def edge_weight_list(self) -> list[float]:
        return self.edge_weight.tolist()

    @property
    def density(self) -> float:
        return self.num_edges / (self.num_nodes * (self.num_nodes - 1))

    @property
    def mean_degree(self) -> float:
        return self.num_edges / self.num_nodes  

    def _validate(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise InvalidGraphStructure(f'Expected edge_index shape [2, num_edges] but got {self.edge_index.shape}')

        if self.edge_weight.ndim != 1:
            raise InvalidGraphStructure(f'Expected edge_weight shape [num_edges] but got {self.edge_weight.shape}')            

        if self.num_edges != self.edge_weight.shape[0]:
            raise InvalidGraphStructure(f"edge_index and edge_weight length mismatch ({self.num_edges}, {self.edge_weight.shape[0]})")

        if self.edge_index.min().item() < 0:
            raise InvalidGraphStructure(f"Node IDs must be non-negative. Got {self.edge_index.min()}")

        if self.edge_index.max().item() >= self.num_nodes:
            raise InvalidGraphStructure(f"Maximum node index is {self.num_nodes}. Starting counting from 0, expected largest to be {self.num_nodes - 1} but got {self.edge_index.max().item()}")

    def _get_adjacency_matrix(self) -> torch.Tensor:
        """returns tensor of adjacency matrix"""
        adj = torch.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=self.edge_weight.dtype,
            device=self.edge_index.device
        )

        adj[self.edge_index[0], self.edge_index[1]] = self.edge_weight

        return adj    
    
    @classmethod
    def from_list(cls, 
                  edge_index:       list[tuple[int, int]], 
                  edge_weight:      list[float],
                  num_nodes:        int) -> Self:
        """
        Returns an instance using Lists as parameters rather than tensors.
        
        Parameters
        ----------
        edge_index: List[Tuple[int, int]]
            shape must be [2, num_edges]
            accepts [num_edges, 2] in addition
        edge_weight: List[float]
            shape must be [num_edges] 
        """
        edge_index_tensor = torch.tensor(edge_index,  dtype = torch.long)
        edge_weight_tensor= torch.tensor(edge_weight, dtype = torch.float)

        if edge_index_tensor.ndim == 2 and edge_index_tensor.shape[1] == 2:
            edge_index_tensor = edge_index_tensor.t()

        return cls(edge_index_tensor, edge_weight_tensor, num_nodes)    
    
    def to(self, device: torch.device) -> "GraphStructure":
        """Move all tensors to the specified device (GPU)"""
        return GraphStructure(
            edge_index = self.edge_index.to(device),
            edge_weight= self.edge_weight.to(device),
            num_nodes  = self.num_nodes
        )        

    def __repr__(self) -> str:
        representation = f'<{self.__class__.__name__}(num_nodes = {self.num_nodes}, num_edges = {self.num_edges})>'
        return representation 