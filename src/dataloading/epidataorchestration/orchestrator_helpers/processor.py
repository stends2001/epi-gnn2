import time
import pandas as pd

from ..utils import NonExistentAttributeEpiDataContainer
from ...epiconfig import EpiConfig
from ..utils.temporal_summary import EpiDataTemporalSummary
from ..containers import ProcessedEpiData, HarmonizedEpiData

class EpiDataProcessor:     
    """
    ``EpiDataOrchestrator`` utility class that creates the ``ProcessedEpiData``. 
    ``EpiDataProcessor`` processes the harmonzied data; transforms cases into incidenc,
    and filters dates.
    
    Besides a handful of helper methods, ``EpiDataProcessor`` has 
    an ``orchestrate()`` method, which returns the ``ProcessedEpiData``.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.    
    temporal_summary : EpiDataTemporalSummary
        Helper class that stores temporal information, built based on ``EpiConfig``.

    Downstream
    ----------        
    ``EpiDataOrchestrator`` has six utility classes, each of which is responsible
    for a single stage in the pipeline of getting model-ready datasets. 
    ``EpiFeatureBuilder`` is the third one.    
    """        
    def __init__(self, 
                 config : EpiConfig, 
                 temporal_summary : EpiDataTemporalSummary):
        
        self.config = config
        self.temporal_summary = temporal_summary

    def _add_incidence_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """adds incidence column"""
        epipopdata['incidence'] = epipopdata['cases'] / epipopdata['population_size'] * self.config.incidence_scalar     

        return epipopdata
    
    def _drop_cases_column(self, epipopdata: pd.DataFrame) -> pd.DataFrame:
        """drops cases column -> only if target => incidence"""

        return epipopdata.drop(columns=['cases']) 
       
    def _filter_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """using the timestamp column, filter on min/max date => using those determined in the temporal summary"""
        
        # Use extended min date from temporal summary
        dfc     = df.copy()
        mindate = self.temporal_summary.get_extended_dates()['min']
        maxdate = self.temporal_summary.get_extended_dates()['max']
             
        dfc = dfc.loc[dfc['timestamp'] <  maxdate].reset_index(drop=True)         
        dfc = dfc.loc[dfc['timestamp'] >= mindate].reset_index(drop=True)     
 

        return dfc
    
    def _filter_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """using the year column, filter on min/max date => using those determined in the temporal summary"""        
        dfc = df.copy() 
        minyear = self.temporal_summary.get_extended_dates()['min'].year

        # for max year its different in year => when max date '2021-02-01' 
        # we still need the data at '2021-01-01' so we can only include year < max_year+1
        maxyear = self.temporal_summary.get_extended_dates()['max'].year + 1

        dfc = dfc.loc[dfc['year'] <  maxyear].reset_index(drop=True)         
        dfc = dfc.loc[dfc['year'] >= minyear].reset_index(drop=True)             

        return dfc

    def orchestrate(self, harmonizeddata: HarmonizedEpiData) -> ProcessedEpiData:
        """Function that orchestrates the others"""
        time_start = time.time()
        epidata = self._add_incidence_column(harmonizeddata.epidata.copy())
        epidata = epidata.drop(columns = 'population_size')

        if self.config.target_column == 'incidence':
            epidata = self._drop_cases_column(epidata)  
              
        epidata = self._filter_dates(epidata)

        # extra features; initiating them on None
        population_size         = None
        population_density_data = None    

        if self.config.feature_popsize:
            if harmonizeddata.population_size is None:
                raise NonExistentAttributeEpiDataContainer(
                    harmonizeddata.__class__.__name__, 'population_size'
                    )

            population_size = self._filter_years(harmonizeddata.population_size)

        if self.config.feature_popdens:
            if harmonizeddata.population_density is None:
                raise NonExistentAttributeEpiDataContainer(
                    harmonizeddata.__class__.__name__, 'population_density'
                    )                

            population_density_data = self._filter_years(harmonizeddata.population_density)
        
        processed_data = ProcessedEpiData(epidata = epidata,   
                                          population_size   = population_size,
                                          population_density= population_density_data
                                          )
        time_end = time.time()

        time_elapsed = time_end - time_start
        return processed_data
