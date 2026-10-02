import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances
import geopandas as gpd 
import pandas as pd
from numpy.typing import NDArray
import numpy as np

from ..utils import GraphType, CRS_METRES
from ...utils import registry_method, get_registered_methods, MethodNotInRegistry

import logging
logger = logging.getLogger(__name__)

class GraphBuilder:
    """ 
    Utiliy - class to ``GraphManager``, responsible for the buidling the raw graph.
    These raw graphs may be processed later. Raw graphs are returned as lists, not 
    tensors. The main orchestration method that calls the others is ``build()``.

    Parameters
    ----------
    id_col : str
        The column name in which the code of each spatial unit is stored. These are the
        ones that will be mapped to node-idx (tokens) alphabetically, the mapping of 
        which will be stored in the overarching directory (``dir_graphs_partition``) 
        under ``tokenization_map.json``.    
    token_col : str
        The column name in which the tokens of the ``id_col`` will be stored.    
    shape_data : gpd.GeoDataFrame
        The shape data of the country at all levels.
    population_data : pd.DataFrame
        The population data of the country at all levels.

    Attributes
    ----------
    ``methods``
        Stores a list of supported building-methods.

    See Also
    --------
    for more information, see ``GraphManager``.
    """
    def __init__(self,
                 id_col : str,
                 token_col : str,                 
                 shape_data : gpd.GeoDataFrame,
                 population_data : pd.DataFrame
                 ):
        
        self.id_col     = id_col 
        self.token_col  = token_col  

        self.shp_data   = shape_data
        self.pop_data   = population_data

        self.methods = get_registered_methods(self.__class__)

    def build(self, method: GraphType, *args, **kwargs) -> tuple[list[tuple[int, int]], list[float]]:
        """
        buidls edge-index and edge-weights according to method. Specific methods may 
        require *args or **kwargs.

        Parameters
        ----------
        method : GraphType
            the method with which to generate the edge indices and edge weights    

        See Also
        --------
        for more information on what parameters are required per method, please see
        the documentation for that method.

        Returns
        -------
        ``edge_index``: List[tuple[int,int]]
            List representation of edges.
        ``edge_weight``: List[float]
            List representation of the weights associated with the edges.
        """
        if method not in self.methods:
            raise MethodNotInRegistry(method, list(self.methods))

        return getattr(self, method)(*args, **kwargs)
    
    @registry_method
    def identity(self) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Build an identity graph; each node is connected to itself only.
        No parameters required.
        """
        node_ids    = list(self.shp_data[self.token_col].unique())
        edges       = [(int(nid), int(nid)) for nid in node_ids]
        weights     = [float(1) for i in range(len(edges))]

        logging.debug('identity graph built')

        return edges, weights

    @registry_method
    def geographical_contiguity(self) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Build a geographical neighbors graph; each node is connected to its 
        geographical neighbors only. No parameters required.
        """        
        neighbors   = gpd.sjoin(self.shp_data, self.shp_data, how='inner', predicate='touches').reset_index(drop=False)
        neighbors   = neighbors[neighbors[f'{self.token_col}_left'] != neighbors[f'{self.token_col}_right']]
        edges       = list(zip(neighbors[f'{self.token_col}_left'], neighbors[f'{self.token_col}_right']))
        edges      += [(t, s) for s, t in edges]
        edges       = list(set(edges))
        weights     = [float(1) for i in range(len(edges))]

        logging.debug('geographical_contiguity graph built')        
        return edges, weights

    @registry_method
    def gravity_model(self,
                      alpha:             float = 2.0,
                      epsilon:           float = 1e-6,
                      decay:             float = 1.0,
                      max_distance:      float = 100_000) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Build a gravity-model based graph: connection strength depends on distance and 
        population size.

        The Gravity formula is as follows:
            edge_weight_{i,j} = pop_i * pop_j / ((distance * decay) ** alpha + epsilon) 
            
            if distance < max_distance else 0

        Parameters
        ----------
        alpha : float = 2
            Distance exponent.
        epsilon : float = 1e-6
            Numerical stability factor (prevents division by 0).
        decay :  float = 1.0
            Higher means stronger decay with distance.
        max_distance : float = 100_000
            Maximum distance between two nodes within which they may still be connected 
            (in m! not in km).
        """
        gdfc             = self.shp_data[[self.token_col, "geometry"]].sort_values(self.token_col).reset_index(drop=True)
        population_data  = self.pop_data.sort_values(self.token_col).reset_index(drop=True)

        dfc_projected               = gdfc.to_crs(CRS_METRES)
        dfc_projected['geometry']   = dfc_projected.geometry.centroid
        coords                      = np.column_stack([dfc_projected.geometry.x, dfc_projected.geometry.y])

        pop                         = population_data["population_size"].to_numpy(dtype=np.float64)
        node_ids                    = population_data[self.token_col].values

        distance_matrix = euclidean_distances(coords)

        # Gravity weights — fully vectorized
        pop_product: NDArray[np.float64]    = np.outer(pop, pop)
        denom                               = (distance_matrix * decay) ** alpha + epsilon
        weight_matrix                       = pop_product / denom

        # Remove self-loops and edges beyond max_distance
        np.fill_diagonal(weight_matrix, 0)
        weight_matrix[distance_matrix > max_distance] = 0

        src, dst = np.nonzero(weight_matrix)
        weights: NDArray[np.float64]  = weight_matrix[src, dst]

        edges_list                      = list(zip(node_ids[src].astype(int).tolist(), node_ids[dst].astype(int).tolist()))
        weights_list: list[float]       = weights.tolist()

        logging.debug('gravity-model graph built')
        return edges_list, weights_list
 
    @registry_method
    def random(self, seed: int = 42, k: int = 5) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Build a random graph where each node has approximately k neighbors.

        Parameters
        ----------
        seed : int = 42
            random seed
        k : int = 5
            target average degree per node
        """
        rng      = np.random.default_rng(seed)
        node_ids = list(self.shp_data[self.token_col].unique())
        n        = len(node_ids)

        # k neighbors out of n-1 possible -> probability per edge
        p = k / (n - 1)

        edges   = []
        weights = []

        for i in node_ids:
            for j in node_ids:
                if i >= j:
                    continue
                if rng.random() < p:
                    edges.append((int(i), int(j)))
                    edges.append((int(j), int(i)))
                    weights.extend([1.0, 1.0])

        logging.debug('random graph built')
        return edges, weights
    
    @registry_method    
    def fully_connected(self) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Build a fully connected graph where each node is connected to all other nodes 
        with weight 1. NOTE: such a graph is computationally expensive!
        """
        node_ids = list(self.shp_data[self.token_col].unique())

        edges = [
            (int(i), int(j))
            for i in node_ids
            for j in node_ids
            if i != j
        ]
        weights = [1.0] * len(edges)

        logging.debug('fully connected graph built')
        return edges, weights    

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}>"
        return representation