from dataclasses import dataclass 
import torch 
from typing import Self, ClassVar
import os 
from pathlib import Path 

from .config import GraphConfig
from .structure import GraphStructure
from ..exceptions import InvalidGraphObject
from ...utils import PathNotFound, load_mapping_dict, save_mapping_dict

import logging
logger = logging.getLogger(__name__)

@dataclass 
class GraphObject:
    """
    Class that stores a ``GraphStructure``, ``GraphConfig`` and a tokenization_map.

    Parameters
    -----------
    graph : GraphStructure
        The pytorch representation of the graph (edge_index and edge_weight).
    tokenization_map : Dict[str,int]
        Mapping of node-identifier (i.e. NUTS code) -> token
    config : GraphConfig   
        config-dataclass for the graph 

    One may also set a graph structure with lists  instead of torch.Tensors, using
    the classmethod `from_list()`. These are then directly converted into Tensors.

    Methods
    ``load()``
        Load ``GraphObject`` from file.
    ``save()``
        Save ``GraphObject`` to file.

    Attributes
    ----------
    ``reverse_tokenization_map``
        Where ``tokenization_map`` is a dict[str, int] (key -> token), this returns
        the opposite; dict[int, str]
    
    See Also
    --------
    ``GraphConfig``
        A vital element of ``GraphObject``. Contains the description on how the 
        ``GraphStructure`` was made.
    ``GraphStructure``
        A vital element of ``GraphObject``. The actual structure for the graph.
    ``GraphRegistry``
        An interactive registry of ``GraphObjects``.        

    Downstream
    ----------
    ``GraphBuilder`` constructs ``GraphStructure``s. In order to keep them organized,
    and reproducible, however, they wrap a ``GraphStructure`` in a ``GraphObject``,
    which contains all information required to recreate the structure. These are
    then saved into ``GraphRegistry``, which is an interactive registry of 
    ``GraphObjects``.
    
    Examples
    ---------
    >>> edges   = [(0,1), (1,0), (3,1)]
    ... graph   = GraphStructure.from_list(edges, [1, 1, 1], 4)    
    ... graphobj= GraphObject(graph, {'A': 0, 'B': 1, 'C': 2, 'D':3}, {})    

    """
    graph:              GraphStructure
    tokenization_map:   dict[str, int]   
    config:             GraphConfig

    # typed as ``ClassVar`` since it's a dataclass.
    edge_index_filename: ClassVar[str] = 'edge_index.pt'
    edge_weight_filename: ClassVar[str] = 'edge_weight.pt'
    graphconfig_filename: ClassVar[str] = 'config.json'
    tokenization_map_filename: ClassVar[str] = 'tokenization_map.json'

    def __post_init__(self):
        self._validate()

    def save(self, path: str | Path) -> None:
        """ 
        Saves a ``GraphObject`` into path, represented by four seperate files:
        ``config.json``
            ``config`` Attribute of ``GraphObject``.
        ``edge_index.pt``
            ``edge_index`` Attribute of ``GraphObject.graph``.
        ``edge_weight.pt``
            ``edge_weight`` Attribute of ``GraphObject.graph``.
        ``tokenization_map.json``
            ``tokenization_map`` Attribute of ``GraphObject``. NOTE: this file is only
            saved in the parent of ``path``, and is only saved if it doesn't exist yet.

        Parameters
        ----------
        path : str | Path
            the path in which to save the ``GraphObject``.
        """

        if isinstance(path, str):
            path = Path(path)

        graphconfig_dict = self.config.asdict()

        if not path.exists():
            path.mkdir()

        save_mapping_dict(graphconfig_dict, path / self.graphconfig_filename)

        torch.save(self.graph.edge_index, path / self.edge_index_filename)
        torch.save(self.graph.edge_weight, path /self.edge_weight_filename)             
        
        if self.tokenization_map_filename not in os.listdir(str(path.parent)):
            save_mapping_dict(self.tokenization_map, path.parent / self.tokenization_map_filename)
            logger.info('Tokenization map saved under %s.', path.parent / self.tokenization_map_filename)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Loads a graph structure into ``GraphObject`` based on supplied path. 

        Returns
        -------
        ``GraphObject``
        """
        if isinstance(path, str):
            path = Path(path)

        # validate path-existence
        if not path.exists():
            raise PathNotFound(path)

        # validate Tokenization map's path
        tokenization_map_path = path.parent / cls.tokenization_map_filename
        if not tokenization_map_path.exists():
            raise PathNotFound(tokenization_map_path)            

        # validate specific files' presence
        for ff in [cls.graphconfig_filename, cls.edge_index_filename, cls.edge_weight_filename]:
            filepath  = path / ff
            if not filepath.exists():
                raise PathNotFound(filepath)                

        graphconfig_dict= load_mapping_dict(path / cls.graphconfig_filename)
        graphconfig     = GraphConfig.fromdict(graphconfig_dict)        

        edge_index      = torch.load(path / cls.edge_index_filename, weights_only=True)
        edge_weight     = torch.load(path / cls.edge_weight_filename, weights_only=True)
        logger.info('Graph %s loaded from file.', path.stem)

        graphstructure  = GraphStructure(edge_index, edge_weight, graphconfig.num_nodes)

        tokenization_map: dict[str, int]= load_mapping_dict(path.parent / cls.tokenization_map_filename)

        return cls(graphstructure, tokenization_map, graphconfig)

    def _validate(self):
        """Briefly validates ``GraphObject``"""
        num_expected_nodes = len(self.tokenization_map.values())
        
        if num_expected_nodes != self.graph.num_nodes:
            raise InvalidGraphObject(f'GraphStructure has {self.graph.num_nodes} nodes but tokenization map has {num_expected_nodes}.')                 
        
        if len(self.tokenization_map.values()) != len(set(self.tokenization_map.values())):
            raise InvalidGraphObject(f'Found doubles inside the values of tokenzation map!')    

    @property 
    def reverse_tokenization_map(self) -> dict[int, str]:
        return  {v: k for k, v in self.tokenization_map.items()}
    
    def __repr__(self) -> str:
        return (
            f"GraphObject(graph={self.graph})"
        )
