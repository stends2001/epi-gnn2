import geopandas as gpd 
import pandas as pd
from pathlib import Path

from ...utils import compare_sets, MissingColumnError, AdminLevel

import logging
logger = logging.getLogger(__name__)

class GraphContextDataProcessor:
    """ 
    Utiliy - class to ``GraphManager``, responsible for the loading and preprocessing of
     context data for the GraphStructure. The main orchestration method that calls the 
     others is ``process()``.

    Parameters
    ----------
    level : Level
        level of the country represented by the graph structure(s) to be created.      
    id_col : str
        The column name in which the code of each spatial unit is stored. These are the
        ones that will be mapped to node-idx (tokens) alphabetically, the mapping of 
        which will be stored in the overarching directory (``dir_graphs_partition``) 
        under ``tokenization_map.json``.    
    token_col : str
        The column name in which the tokens of the ``id_col`` will be stored.    
    country_path : Path   
        The path in which the data for this country can be found.
      
    See Also
    --------
    for more information, see ``GraphManager``.
    """
    def __init__(self,
                 level: AdminLevel,
                 id_col : str,
                 token_col : str,
                 country_data_path : Path
                 ):
        
        self.id_col             = id_col 
        self.token_col          = token_col  
        self.level              = level  
        self.country_data_path  = country_data_path

        # load raw data
        self.shp_raw, self.pop_raw = self._load()

        # validate input
        self._validate()

    def process(self) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, int]]:
        """
        processes raw data and returns processed data; shapedata and populationdata
        interanlly calls `_filter()` and `_tokenize()`
        """
        shp_f, pop_f = self._filter(self.shp_raw, self.pop_raw)
        shp_t, pop_t, map_t = self._tokenize(shp_f, pop_f)

        logging.debug('data processed')
        return shp_t, pop_t, map_t

    # ============= HIDDEN METHODS ============ #
    def _load(self) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """loads and returns shapedata and population data"""
        shp = gpd.read_file(self.country_data_path / 'level_shapes.shp')
        pop = pd.read_csv(self.country_data_path  / 'population_size.csv', dtype = {'key': 'str'})        
        logging.debug('data loaded')
        return shp, pop

    def _validate(self):
        """internally called validation - method. Checks for id-columns' presence among the datasets."""
        if self.id_col not in self.shp_raw.columns:
            raise MissingColumnError(self.id_col, 'shape data')
        
        if self.id_col not in self.pop_raw.columns:
            raise MissingColumnError(self.id_col, 'population data')
        
        logging.debug('data validated')        

    def _filter(self,
                 shape_data: gpd.GeoDataFrame, 
                 population_data: pd.DataFrame)  -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """filters data on level and year"""
        shp_f = shape_data[shape_data['level'] == self.level].reset_index(drop=True)
        pop_f = population_data[population_data['level'] == self.level].reset_index(drop=True)
        pop_f = pop_f[pop_f['year'] == 2020].reset_index(drop = True)
        
        logging.debug('data filtered')            
        return shp_f, pop_f

    def _tokenize(self, 
                 shape_data: gpd.GeoDataFrame, 
                 population_data: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, int]]:
        """sets a tokenization_map and tokenizes all DataFrames"""
        
        shape_data              = shape_data.copy()
        population_data         = population_data.copy()    

        unique_shape_data_ids   = set(shape_data[self.id_col])
        tokenization_map        = {str(key): value for value, key in enumerate(sorted(unique_shape_data_ids))}        

        if population_data is not None:
            unique_population_data_ids = set(population_data[self.id_col])

            compare_sets(unique_shape_data_ids,unique_population_data_ids)

            population_data[self.token_col] = (
                population_data[self.id_col]
                .map(tokenization_map)
                .astype(int)
            )

            if not population_data[self.token_col].notna().all():
                raise ValueError(f'Unmapped keys found in population data:  {population_data[self.id_col][population_data[self.token_col].isna()].tolist()}')

            population_data.drop(columns = self.id_col, inplace = True)

        shape_data[self.token_col] = (
            shape_data[self.id_col]
                .map(tokenization_map)
                .astype(int)
        )    

        if not shape_data[self.token_col].notna().all():
            raise ValueError(f'Unmapped keys found in shape data:  {shape_data[self.id_col][shape_data[self.token_col].isna()].tolist()}')            

        shape_data.drop(columns = self.id_col, inplace = True)            

        logging.debug('data tokenized')        

        return shape_data, population_data, tokenization_map
    
    def __repr__(self) -> str:
        representation = f'<{self.__class__.__name__}>'
        return representation