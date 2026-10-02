import pandas as pd 
from dataclasses import dataclass

from ....utils.textformatting import checkmark

@dataclass
class HarmonizedEpiData:
    """
    Data container during data-orchestration for stage 2: harmonized data.
    At this stage, data is aggregated onto the right administrative level, and 
    the regions are now tokenized. The identification - column (name defined by 
    ``EpiConfig``) is of type integer.

    Parameters
    ----------
    epidata : pd.DataFrame
        Data related to epidemiology at the right level. Expected columns are:
        - ``'timestamp'``
        - ``'cases'``
        - ``'year'``
        - ``'population_size'``
        - ``{epiconfig.id_column}``
    population_size : pd.DataFrame | None = None
        Optional population size data at the right level. Only expected to be filled
        when ``feature_popsize`` of ``EpiConfig`` is set to ``True``. Expected columns 
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
    ``EpiDataHarmonizer``
        Loads harmonized data into ``HarmonizedEpiData``.
    ``HarmonizedValidator``        
        Validates the integrity of ``HarmonizedEpiData``.        

    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. ``HarmonizedEpiData`` is the second of these.
    """      
    epidata : pd.DataFrame

    population_size : pd.DataFrame | None = None
    population_density : pd.DataFrame | None = None
       

    def __repr__(self):
        representation = f"<{self.__class__.__name__}(epidata {checkmark}"

        if self.population_size is not None:
            representation += f", population_size {checkmark}"

        if self.population_density is not None:
            representation += f", population_density {checkmark}"

        representation += ")>"
        return representation
