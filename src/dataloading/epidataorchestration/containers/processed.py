import pandas as pd 
from dataclasses import dataclass

from ....utils.textformatting import checkmark

@dataclass
class ProcessedEpiData:
    """
    Data container during data-orchestration for stage 3: processed data.

    Parameters
    ----------
    epidata : pd.DataFrame
        Processed epidemiological data. Expected columns are:
        - ``{epiconfig.temporal_column}``
        - ``'year'``
        - ``{epiconfig.id_column}``
        - ``{epiconfig.target_column}``        
    population_size : pd.DataFrame | None = None
        Optional population size data at the right level. Only expected to be filled
        when ``feature_popsize`` of ``EpiConfig`` is set to ``True``.  Expected columns 
        are:
        - ``'year'``
        - ``'population_size'``
        - ``{epiconfig.id_column}``
    population_density: pd.DataFrame | None = None
        Optional population density data at the right level. Only expected to be filled
        when ``feature_popdens`` of ``EpiConfig`` is set to ``True``.  Expected columns 
        are:
        - ``'year'``
        - ``'population_density'``
        - ``{epiconfig.id_column}``
    
    See Also
    --------
    ``EpiDataProcessor``
        Loads context data into ``ProcessedEpiData``.
    ``ProcessedValidator``        
        Validates the integrity of ``ProcessedEpiData``.        
        
    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. ``ProcessedEpiData`` is the third of these.
    """             
    epidata : pd.DataFrame

    population_size : pd.DataFrame | None = None
    population_density : pd.DataFrame | None = None
       
             

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(epidata {checkmark}")

        if self.population_size is not None:
            representation += f", population_size {checkmark}"

        if self.population_density is not None:
            representation += f", population_density {checkmark}"

        representation += ")>"
        return representation