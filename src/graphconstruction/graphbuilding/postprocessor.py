import torch
from typing import Literal, assert_never

from ..graphobjects import GraphStructure
from ...utils import registry_method, get_registered_methods, MethodNotInRegistry
from ..utils import GraphNormType

import logging
logger = logging.getLogger(__name__)

class GraphPostProcessor:
    """ 
    Utiliy - class to ``GraphManager``, responsible for the processing of graphs by
    - (optional) top-k - filtering
    - edge-weight normalization

    The main orchestration method that calls the others is ``normalize()``.
      
    Attributes
    ----------
    ``methods``
        Stores a list of supported normalization-methods        

    See Also
    --------
    For more information, see ``GraphManager``.
    """

    def __init__(self):
        self.methods     = get_registered_methods(self.__class__)

    def filter_top_k(self, 
                     graph_structure: GraphStructure, 
                     mode: Literal['local','global'],
                     k: int
                     ) -> GraphStructure:
        """
        top_k filtering: selecting the `k` most weighted edges, either
        globally or locally (per neighborhood/node).
        
        Parameters
        ----------
        graph_structure: GraphStructure
            graph structure to which edges are to be filtered
        mode: Literal['local','global']
            to determine the number of connections over the entire graph or per node
        k: int
            the number of connections to keep

        Returns
        -------
        Graphstructure
        """
        edge_index   = graph_structure.edge_index
        edge_weight  = graph_structure.edge_weight

        # determine the edge indices to keep
        match mode:

            case "global":
                keep = edge_weight.topk(k).indices

            case "local":
                keep = []

                for node in range(graph_structure.num_nodes):
                    mask = edge_index[0] == node
                    node_edges = mask.nonzero(as_tuple=True)[0]

                    if len(node_edges) <= k:
                        keep.append(node_edges)
                    else:
                        top = edge_weight[node_edges].topk(k).indices
                        keep.append(node_edges[top])

                keep = torch.cat(keep)

            case _:
                assert_never(mode)

        # filter on these edges
        edge_index_f  = edge_index[:, keep]
        edge_weight_f = edge_weight[keep]

        filtered_graphstructure = GraphStructure(
            edge_index_f,
            edge_weight_f,
            graph_structure.num_nodes,
        )
        logging.debug('topk filtered')
        return filtered_graphstructure

    def normalize(self, graph_structure: GraphStructure, method: GraphNormType, *args, **kwargs) -> GraphStructure:
        """ 
        Normalizes edge-weights according to method. Specific methods may require 
        *args or **kwargs.    

        Parameters
        ----------
        graph_structure : GraphStructure
            The graph structure to be normalized.
        method:  GraphNormType
            The method with which to normalize the edge weights.

        Returns
        -------
        ``GraphStructure``
            This a full-blown, normalized graph structure.            

        See Also
        --------
        For more information, see ``GraphManager``.

        For more information on what parameters are required per method, please see
        the documentation for that method.
        """        
        if method not in self.methods:
            raise MethodNotInRegistry(method, list(self.methods))
    
        normalized_weights: torch.Tensor = getattr(self, method)(graph_structure, *args, **kwargs)

        normalized_graph_structure = GraphStructure(
            graph_structure.edge_index,
            normalized_weights,
            graph_structure.num_nodes,
        )

        return normalized_graph_structure

    @registry_method
    def none(self,
             graph_structure: GraphStructure) -> torch.Tensor:
        """ 
        does not normalize edge weights. returns the tensor of the input weights.
        """
        logging.debug('not normalized')
        return graph_structure.edge_weight 

    @registry_method
    def minmax(self, 
               graph_structure: GraphStructure, 
               epsilon: float = 1e-3) -> torch.Tensor:
        """
        scales edge-weights by minmax (i.e. max => 1 and min => 0 + `epsilon`)

        Parameters
        ----------
        epsilon: float = 1e-3
            the mininum value
        """
        raw_weights = graph_structure.edge_weight

        min_w = raw_weights.min().item()
        max_w = raw_weights.max().item()    

        if max_w > min_w:
            edge_weights = (epsilon + raw_weights - min_w) / (max_w - min_w)
        else:
            edge_weights = torch.zeros_like(raw_weights)    
        logging.debug('minmax-normalized')
        return edge_weights       

    @registry_method
    def symmetric(self, graph_structure: GraphStructure) -> torch.Tensor:
        """
        Applies symmetric normalization to edge weights.
        
        Further Reading
        ---------------
        This should be the default normalization method

        This computes the normalized edge weights according to:

            w_ij' = w_ij / sqrt(d_i * d_j)

        where:
            - w_ij is the original edge weight between node i -> j
            - d_i is the (weighted) out-degree of node i:
                d_i = sum_j w_ij

        In matrix form, this corresponds to:

            D^{-1/2} A D^{-1/2}

        where:
            - A is the weighted adjacency matrix
            - D is the diagonal degree matrix
        """
        indices     = graph_structure.edge_index
        raw_weights = graph_structure.edge_weight
        row, col    = indices[0], indices[1]

        deg = torch.zeros(graph_structure.num_nodes, dtype=raw_weights.dtype, device=raw_weights.device)
        deg.scatter_add_(0, row, raw_weights)

        deg_inv_sqrt                               = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = raw_weights * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        logging.debug('symmetrically-normalized')        
        return edge_weight

    @registry_method
    def rowwise(self, graph_structure: GraphStructure) -> torch.Tensor:
        """
        Applies row-wise normalization: D^(-1) * W
        Each node's outgoing edge weights sum to 1.
        """
        indices     = graph_structure.edge_index
        raw_weights = graph_structure.edge_weight
        row, col    = indices[0], indices[1]

        deg = torch.zeros(graph_structure.num_nodes).to(raw_weights)
        deg.scatter_add_(0, row, raw_weights)

        deg_inv                          = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0

        edge_weight = raw_weights * deg_inv[row]
        logging.debug('rowwise-normalized')        
        return edge_weight
    
    def __repr__(self) -> str:
        representation = f'<{self.__class__.__name__}>'
        return representation