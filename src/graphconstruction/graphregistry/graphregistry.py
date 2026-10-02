from pathlib import Path 

from ..graphobjects import GraphObject
from ...utils import PathNotFound

import logging
logger = logging.getLogger(__name__)

class InvalidGraphName(Exception):
    def __init__(self, graphname: str, supported_chars: set[str]):
        msg = f"Graphname '{graphname}' invalid. Supported characters are: {supported_chars}"
        super().__init__(msg)

class GraphRegistry:
    """
    Registry of ``GraphObject`` instances: stores, saves, loads and interacts with them.

    Parameters
    ----------
    ``graph_partition_dir`` : str
        partition-directory in which to save graphstructures from the registry. That is,
        the path in which different graph representations of the same thing are stored 
        (same country/level combination). These structures have a shared 
       ``tokenization_map``.

    Methods
    -------
    ``add_entry()``
        Add a ``GraphObject`` to ``GraphRegistry``.
    ``get_entry()``
        Get a ``GraphObject`` from ``GraphRegistry``.    
    ``rename_entry()``
        Rename a ``GraphObject`` in ``GraphRegistry``.    
    ``remove_entry()``
        Remove a ``GraphObject`` from ``GraphRegistry``.      
    ``save_entry()``
        Save a ``GraphObject`` from ``GraphRegistry`` in ``graph_partition_dir``.      
    ``save_all_entries()``
        Save all ``GraphObject``s from ``GraphRegistry`` in ``graph_partition_dir``.          
    ``load_entry()``
        Load a saved ``GraphObject`` into ``GraphRegistry``.    

    Attributes
    ----------
    ``entry_names``
        List of names of all ``GraphObject``s in ``GraphRegistry``.

    See Also
    --------
    ``GraphObject``
        The class in which graphs are wrapped.
    ``GraphStrucure``
        The ``graph`` attribute of ``GrahObject``.

    Downstream Use
    ---------------
    Each instance of ``GraphManager`` has a registry as attribute. These create new
    graph structures, wraps them in ``GraphObject``s, and stores these into 
    ``GraphRegistry``.
    """
    def __init__(self, graph_partition_dir: str | Path):
        self._registry: dict[str, GraphObject]= {}

        graph_partition_dir = Path(graph_partition_dir)
        
        if not graph_partition_dir.exists():
            raise PathNotFound(graph_partition_dir)

        self.graph_partition_dir = graph_partition_dir

    def add_entry(self, graphname: str, entry: GraphObject) -> None:
        """adds entry to registry under graphname"""
        if self._check_presence(graphname):
            logger.warning("Graph '%s' already exists in registry; new graph not added.", graphname)           

        else:
            self._registry[graphname] = entry
            logger.info("Graph '%s' registered.", graphname)
        
    def get_entry(self, graphname: str) -> GraphObject | None:
        """returns entry from graphname"""
        if not self._check_presence(graphname):
            logger.warning("Graph '%s' could not be found in registry.", graphname)
        
        else:
            return self._registry[graphname]

    def rename_entry(self, current_graphname: str, new_graphname: str) -> None:
        """renames entry from current_graphname to new_graphname; current_graphname is removed"""        
        if not self._check_presence(current_graphname):
            logger.warning("Graph '%s' could not be found in registry.", current_graphname)
            return
        
        if self._check_presence(new_graphname):
            logger.warning("Graph '%s' already exists in registry; new graph not added.", new_graphname)
            return 

        self.add_entry(new_graphname, self._registry[current_graphname])
        self.remove_entry(current_graphname)          

    def remove_entry(self, graphname) -> None:
        """removes entry from registry"""
        if not self._check_presence(graphname):
            logger.warning("Graph '%s' could not be found in registry.", graphname)
            return 
        
        del self._registry[graphname]
        logger.info("Graph '%s' removed.", graphname)

    def save_entry(self, graphname: str) -> None:
        """save a graph entry from registry to `self.graph_partition_dir` / `graphname` """
        entry_to_save = self.get_entry(graphname)
        
        if entry_to_save is None:
            # no need to log, already done in `get_entry()`
            return

        self._validate_graphentry_name(graphname)
        
        # save into partition (country / level) / graphname
        entry_to_save.save(self.graph_partition_dir / graphname)

        logger.info("Graph '%s' saved into path %s.", graphname, self.graph_partition_dir / graphname)

    def save_all_entries(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError(f'when saving all graph structures, please confirm by argument')
        
        for entryname in self.entry_names:
            self.save_entry(entryname)

    def load_entry(self, graphname: str) -> None:
        """load a graph entry from directory"""       

        # load from partition (country / level) / graphname        
        graph_object = GraphObject.load(self.graph_partition_dir / graphname)

        self.add_entry(graphname, graph_object)

    @property 
    def entry_names(self) -> list[str]:
        return list(self._registry.keys())
    
    # ============ HIDDEN METHODS ================ #
    def _check_presence(self, graphname: str) -> bool:
        return graphname in self._registry.keys()

    def _validate_graphentry_name(self, graphname: str) -> None:
        """test whether name is suitable or not to be saved into a directory"""

        characters_allowed =  set("abcdefghijklmnopqrstuvwxyz0123456789_-")

        if any(ch not in characters_allowed for ch in graphname):
            raise InvalidGraphName(graphname, characters_allowed)

    def __repr__(self) -> str:
        registered_entries = ', '.join(self.entry_names)
        return f'<{self.__class__.__name__}(N = {len(self.entry_names)} graphs:\n\t{registered_entries})>'