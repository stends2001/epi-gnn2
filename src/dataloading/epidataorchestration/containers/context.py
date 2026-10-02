import pandas as pd 
import geopandas as gpd
from dataclasses import dataclass

from ..utils import EpiDataTemporalSummary
from ....utils import Country, AdminLevel
from ....utils.textformatting import checkmark


@dataclass 
class ContextEpiData:
    """
    Data container during data-orchestration for stage 2.5: context data.

    Parameters
    ----------
    country : Country
        The country in the experiment. Taken from ``country`` attribute of 
        ``EpiConfig``.
    level : AdminLevel
        The administrative level in the experiment. Taken from ``level`` attribute of 
        ``EpiConfig``.
    global_shapedata : gpd.GeoDataFrame
        The original shape data of the country. Expected columns are: 
        - ``'level'``
        - ``'key'``
        - ``'geometry'``
    local_shapedata : gpd.GeoDataFrame
        The shape data for the tokenized nodes. This is the shapedata of the specific
        level with tokenized identities for the regions. These are stored in column 
        ``{id_column}`` attribute of ``EpiConfig``. Expected columns are:
        - ``'geometry'``
        - ``{epiconfig.id_column}``
    population_size : pd.DataFrame
        The pouplation size data for the tokenized nodes. Expected columns are
        - ``'year'``
        - ``'population_size'``
        - ``{epiconfig.id_column}``
    nodenames : pd.DataFrame
        The name of each region for the country/level - combination. Expected columns 
        are:
        - ``'key'``
        - ``f'{epiconfig.level}_name'``
        - ``{epiconfig.id_column}``
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
    temporal_summary : EpiDataTemporalSummary
        Helper class that stores temporal information, built based on ``EpiConfig``.

    See Also
    --------
    ``EpiDataHarmonizer``
        Loads context data into ``ContextEpiData``.
    ``ContextValidator``        
        Validates the integrity of ``ContextEpiData``.        
    ``EpiDataTemporalSummary``
        Helper class that stores temporal information, built based on ``EpiConfig``.
        
    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. ``ContextEpiData`` is the second(.5) of these.

    Raw data is processed into harmonized data, stored in ``HarmonizedEpiData``. The 
    data related, but not necessarily needed in the rest of the orchestration-pipeline
    is stored in this ``ContextEpiData``.
    """    
    country : Country
    level : AdminLevel
    global_shapedata : gpd.GeoDataFrame   
    local_shapedata : gpd.GeoDataFrame     
    population_size : pd.DataFrame
    nodenames : pd.DataFrame
    region_harmonization : pd.DataFrame
    tokenization_map : dict[str, int]
    temporal_summary : 'EpiDataTemporalSummary'

    @property
    def num_nodes(self) -> int:
        return len(self.local_shapedata)

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(country = {self.country}, "
                f"level = {self.level}, "
                f"global_shapedata {checkmark}, "             
                f"local_shapedata {checkmark}, "                  
                f"population_size {checkmark}, "                   
                f"region_harmonization {checkmark}, "                                                                      
                f"nodenames {checkmark}, "
                f"tokenization_map {checkmark}, "
                f"temporal_summary {checkmark}"            
                )
        representation += ")>"
        return representation