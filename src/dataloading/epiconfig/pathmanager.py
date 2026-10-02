from typing import Callable
from pathlib import Path
import inspect

from ...utils import PathManager, Country, AdminLevel, Disease

def registered_property(func: Callable) -> Callable:
    """
    Marks a property and ensures the returned path exists.
    This decorator must be used BEFORE ``@property``.
    """
    def wrapper(self):
        path = func(self)

        if not isinstance(path, Path):
            raise TypeError(f"{func.__name__} did not return a Path")

        return path

    setattr(wrapper, '_is_path', True)
    return wrapper

def get_registered_properties(cls: type) -> list[str]:
    """
    Returns names of all properties decorated with ``@registered_property``.
    """
    return [
        name
        for name, attr in inspect.getmembers(cls)
        if isinstance(attr, property) and getattr(attr.fget, '_is_path', False)
    ]

class EpiPathsManager:
    """
    Stores paths relevant to ``EpiConfig``. 

    Parameters
    ----------
    country : Country
        The country fow which data needs to be retrieved.
    level : AdminLevel
        The admininstrative level for which data needs to be retrieved.
    disease: Disease
        The disease for which case data needs to be retrieved.

    Attributes
    ----------
    ``properties`` 
        A list of all paths stored (in properties).

    Methods
    -------
    ``get()``
        Get the path of a property-name.    

    See Also
    --------
    ``EpiConfig``
        The class to which ``EpiPathsManager`` is a helper class.
    ``EpiValidator``
        The class that validates the paths stored in ``EpiPathsManager`` by accessing
        ``.properties``.

    Downstream
    ----------
    ``EpiConfig`` stores all configuration information needed to transform raw data
    into model-ready datasets by ``EpiDataOrchestrator``. ``EpiValidator`` validates the
    input to ``EpiConfig``, as well as the paths that ``EpiPathsManager`` stores.
    """
    def __init__(self, 
                 country : Country,
                 level : AdminLevel,
                 disease : Disease):
        
        self.properties = get_registered_properties(self.__class__)
        self.country    = country
        self.level      = level
        self.disease    = disease
        self.pm         = PathManager()

    def get(self, property: str) -> Path:
        if property not in self.properties:
            raise ValueError(f'{property} is not a known property of {self.__class__.__name__}. Valid Properties are {self.properties}')
        
        return getattr(self, property)

    # ======= PATHS SHARED AMONG PATHMANAGERS ====== #        
    @property
    @registered_property    
    def data_env(self) -> Path:
        return self.pm.data 

    @property
    @registered_property    
    def population_size(self) -> Path:
        """Path to population size CSV file."""
        return self.data_env / f'{self.country}/population_size.csv'  

    @property
    @registered_property   
    def region_harmonization(self) -> Path:
        """Path to NUTS names file."""
        return self.data_env / f'{self.country}/level_harmonization.tsv'

    @property
    @registered_property   
    def shapefile(self) -> Path:    
        """Path to shapefile of the country (including all levels)"""
        return self.data_env / f'{self.country}/level_shapes.shp'        
    
    @property
    @registered_property   
    def tokenization_map(self) -> Path:    
        """Path to shapefile of the country (including all levels)"""
        return self.data_env / f'{self.country}/graphs/{self.level}/tokenization_map.json'       
    
    @property
    @registered_property   
    def cases(self) -> Path:
        """Path to disease CSV file."""
        return self.data_env / f'{self.country}'/  f'{self.disease}.csv'
    
    @property
    @registered_property     
    def population_density(self) -> Path:
        """Path to population density CSV file."""
        return self.data_env / f'{self.country}/population_density.csv'         

    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}>"
        return representation        
