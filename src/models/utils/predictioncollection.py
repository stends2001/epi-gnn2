import pandas as pd 
from dataclasses import dataclass, field

from .exceptions import MissingPredictionsError

@dataclass
class PredictionCollection:
    """
    Stores predictions across horizons for a single datast (train/val/test).
    
    Predictions are stored in a dictionary with three levels:

    - ``horizon`` : int (counting from zero)
    - ``is_original`` : bool (original scale (True) or transformed scale (False))
    - ``spatially_aggregated`` : bool (national (True) or per spatial unit (False))
    
    Methods
    -------
    ``add()``
        Add a dataset to ``PredictionCollection``.
    ``get()``
        Get a dataset from ``PredictionCollection``.    

    See Also
    --------
    ``PredictionManager``
        Stores three ``PredictionCollection``s, one for each of train/val/test.

    Downstream
    ----------
    Each mode class is a subclass of ``BaseModel``, which creates an instance of 
    ``PredictionManager``. These in turn consist of three ``PredictionCollection``s.
    """

    # a new dictionary is created for each class' instance
    _data: dict[tuple[int, bool, bool], pd.DataFrame] = field(default_factory=dict)      
    
    def add(self, 
            data : pd.DataFrame, 
            horizon : int, 
            is_original : bool, 
            spatially_aggregated : bool):
        """
        Add predictions to storage.

        Parameters
        ----------
        data : pd.DataFrame
            The model's predictions for the specific combination of parameters.
        horizon : int
            The horizon to store. Starts at zero, irrespective of horizon lead time.
        is_original : bool
            If ``False``, then transformed scale, if ``True`` then original scale.        
        spatially_aggregated : bool
            Whether the predictions are per node (``False``), or spatially aggregated 
            (i.e. national) (``True``).
        """
        self._data[(horizon, is_original, spatially_aggregated)] = data
    
    def get(self, 
            horizon : int, 
            is_original : bool, 
            spatially_aggregated : bool) -> pd.DataFrame:
        """
        Get predictions from storage.

        Parameters
        ----------
        horizon : int
            The horizon to store. Starts at zero, irrespective of horizon lead time.
        is_original : bool
            If ``False``, then transformed scale, if ``True`` then original scale.             
        spatially_aggregated : bool
            Whether the predictions are per node (``False``), or spatially aggregated 
            (i.e. national) (``True``).
        """
        key = (horizon, is_original, spatially_aggregated)

        if key not in self._data:
            raise MissingPredictionsError(
                f"No predictions found for horizon={horizon}, is_original={is_original}, spatially_aggregated={spatially_aggregated}. Available: {list(self._data.keys())}"
                )
        
        return self._data[key].copy()

    @property
    def horizons(self) -> list[int]:
        """return a list of horizon integers for which predictions are found"""
        return sorted(set(h for h, _, _ in self._data.keys()))
    
    def _contains_data(self) -> bool:
        """return bool for whether or not predictions exist"""
        return bool(self.horizons)
    
    def __repr__(self) -> str:
        if self._contains_data():
            representation =  f"predictions for horizons {self.horizons}"
        else:
            representation = "no predictions"
        
        return f"<{self.__class__.__name__}({representation})>"