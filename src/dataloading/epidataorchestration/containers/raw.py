import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass

from ....utils.textformatting import checkmark

@dataclass
class RawEpiData:
    """
    Data container during data-orchestration for stage 1: raw data.

    Parameters
    ----------
    disease : pd.DataFrame
        Raw case data per regional unit, per time step.
    population_size : pd.DataFrame
        Population size covering the country over different levels. Expected columns
        are: 
        - ``'level'``
        - ``'key'``
        - ``'year'``
        - ``'population_size'``
    shapedata : gpd.GeoDataFrame
        Shapedata covering the country over different levels. Expected columns are:
        - ``'level'``
        - ``'key'``
        - ``'geometry'``
    region_harmonization : pd.DataFrame
        Data frame that covers which NUTS3 units belong to which NUTS2 and NUTS1
        regions. Expected columns are:
        - ``'nuts3_key'``
        - ``'nuts2_key'``
        - ``'nuts1_key'``
        - ``'nuts3_name'``
        - ``'nuts2_name'``
        - ``'nuts1_name'``
    tokenization_map : dict[str, int]
        Tokenization map used for the graph structures. For each country/level 
        combination (e.g. Germany NUTS3), there should be a tokenization_map.json saved,
        created by the ``GraphConstruction`` - module.
    population_density: pd.DataFrame | None = None
        Optional population density covering the country over different levels. 
        This attribute is only filled when the attribute ``feature_popdens`` of 
        ``EpiConfig`` is set to ``True``. Expected columns are:
        - ``'level'``
        - ``'key'``
        - ``'year'``
        - ``'population_density'``

    See Also
    --------
    ``EpiDataReader``
        Loads raw data into ``RawEpiData``.
    ``RawValidator``        
        Validates the integrity of ``RawEpiData``.

    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. ``RawEpiData`` is the first of these.
    """

    disease : pd.DataFrame
    population_size : pd.DataFrame
    shapedata : gpd.GeoDataFrame
    region_harmonization : pd.DataFrame    
    tokenization_map : dict[str, int]
    
    population_density : pd.DataFrame | None = None        

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(disease {checkmark}, "
                f"population_size {checkmark}, "
                f"shapedata {checkmark}, "               
                f"region_harmonization {checkmark}, "
                f"tokenization_map {checkmark}"                )
        
        if self.population_density is not None:
            representation += f", population_density {checkmark}"              

        return representation +")>"