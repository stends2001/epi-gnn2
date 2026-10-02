import time
import json
from typing import assert_never
import pandas as pd
import geopandas as gpd

from ...epiconfig import EpiConfig
from ..containers import RawEpiData

class EpiDataReader:
    """
    ``EpiDataOrchestrator`` utility class that creates the ``RawEpiData``.
    Besides a handful of helper methods, ``EpiDataReader`` has an ``orchestrate()`` 
    method, which returns the ``RawEpiData``.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.

    See Also
    --------
    ``EpiPathsManager``
        Collects the paths of data to be retrieved by ``EpiDataReader``.

    Downstream
    --------
    ``EpiDataOrchestrator`` has six utility classes, each of which is responsible
    for a single stage in the pipeline of getting model-ready datasets. 
    ``EpiDataReader`` is the first one.
    """
    
    def __init__(self, epiconfig : EpiConfig):
        self.epiconfig = epiconfig

    def orchestrate(self) -> RawEpiData:
        time_start = time.time()

        # create raw data object
        rawdata = RawEpiData(
            disease             = self._load_disease_data(),
            population_size     = self._load_population_size_data(),
            shapedata           = self._load_shapedata(),
            region_harmonization= self._load_regional_harm(),
            tokenization_map    = self._load_tokenization_map(),
        
            # optional data
            population_density  = self._load_population_density() if self.epiconfig.feature_popdens else None         
        )

        time_end = time.time()
        time_elapsed = time_end - time_start
        
        return rawdata

    def _load_disease_data(self) -> pd.DataFrame:
        """
        loads disease data from path in ``EpiPathsManager``. Expected columns are:
        - ``'timestamp'``
        - ``'nuts3_key'``
        - ``'cases'``

        Returns
        -------
        pd.DataFrame
            Raw case data per week per NUTS3 - region, with the timestamp read as 
            TimeDelta. 
        """        
        filepath    = self.epiconfig.path_manager.get('cases')

        match self.epiconfig.country:
            case 'germany':
                initial_key  = 'kz_kreis'
                renamed_key  = 'nuts3_key' 

            case 'hungary':
                initial_key  = 'nuts3_key'
                renamed_key  = 'nuts3_key'                
            
            case _:
                assert_never(self.epiconfig.country)

        df = pd.read_csv(
            filepath,
            parse_dates = ['timestamp'],
            dtype       = {initial_key:  str, 
                           'cases':      int}
        ).rename(columns={initial_key: renamed_key})    
        return df
    
    def _load_population_size_data(self) -> pd.DataFrame:
        """
        loads population size data from path in ``EpiPathsManager``. 
        Expected columns are:
        - ``'level'``
        - ``'key'``
        - ``'year'``
        - ``'population_size'``

        Returns
        -------
        pd.DataFrame
            Raw population size data per year per level, per spatial unit. 
        """         
        filepath = self.epiconfig.path_manager.get('population_size')
        
        df = pd.read_csv(
            filepath,
            dtype = {'key' : str}
        )
        return df

    def _load_shapedata(self) -> gpd.GeoDataFrame:
        """
        loads shapedata for the entire country, with every level and every unit.
        Expected columns are:
        - ``'level'``
        - ``'key'``
        - ``'geometry'``

        Returns
        -------
        gpd.GeoDataFrame
            Raw shape data per year per level, per spatial unit.         
        """          
        filepath = self.epiconfig.path_manager.get('shapefile')
        
        gdf             = gpd.read_file(filepath) # type: ignore
        gdf['key']      = gdf['key'].astype(str)
        return gdf
      
    def _load_regional_harm(self) -> pd.DataFrame:
        """
        Loads regional harmonization data from path in ``EpiPathsManager``. This data 
        frame covers which NUTS3 units belong to which NUTS2 and NUTS1 regions. 
        Expected columns are:
        - ``'nuts3_key'``
        - ``'nuts2_key'``
        - ``'nuts1_key'``
        - ``'nuts3_name'``
        - ``'nuts2_name'``
        - ``'nuts1_name'``

        Returns
        -------
        pd.DataFrame
            Raw regional harmonization data      
        """  

        filepath = self.epiconfig.path_manager.get('region_harmonization')   
        
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        return df

    def _load_tokenization_map(self) -> dict[str, int]:
        """
        Loads the tokenization map used for the graph structures. For each country/level 
        combination (e.g. Germany NUTS3), there should be a tokenization_map.json saved,
        created by the ``GraphConstruction`` - module.        

        Returns
        -------
        dict[str, int]
            Tokenization map where keys correspond to the original keys at this 
            administrative level and values to the integer of the token.
        """
        
        filepath = self.epiconfig.path_manager.get('tokenization_map')       

        with open(filepath, "r") as f:
            tokenization_map = json.load(f)
        return tokenization_map   

    def _load_population_density(self) -> pd.DataFrame:
        """
        loads population density data from path in ``EpiPathsManager``. 
        Expected columns are:
        - ``'level'``
        - ``'key'``
        - ``'year'``
        - ``'population_density'``

        Returns
        -------
        pd.DataFrame
            Raw population density data per year per level, per spatial unit. 
        """              
        filepath = self.epiconfig.path_manager.get('population_density')
        
        df = pd.read_csv(
            filepath,
            dtype   = {'key': str},
        )
                
        return df
