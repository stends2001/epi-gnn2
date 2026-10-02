from __future__ import annotations

import time
from typing import Literal
import pandas as pd
import geopandas as gpd

from ...epiconfig import EpiConfig
from ..utils import berlin_district_ids, berlin_id, EpiDataTemporalSummary
from ..containers import RawEpiData, HarmonizedEpiData, ContextEpiData

# ============= HARMONIZATION CLASS =============
class EpiDataHarmonizer:
    """
    ``EpiDataOrchestrator`` utility class that creates the ``HarmonizedEpiData`` and
    ``ContextEpiData``. ``EpiDataHarmonizer`` aggregates data to the right geograhpical
    level, and tokenizes regions. Further also does the resampling concerning temporal
    resolution. 
    
    Besides a handful of helper methods, ``EpiDataHarmonizer`` has 
    an ``orchestrate()`` method, which returns the ``HarmonizedEpiData`` and 
    ``ContextEpiData``.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.

    See Also
    --------
    ``EpiDataTemporalSummary``
        Helper class that stores temporal information, built based on ``EpiConfig``.

    Downstream
    --------
    ``EpiDataOrchestrator`` has six utility classes, each of which is responsible
    for a single stage in the pipeline of getting model-ready datasets. 
    ``EpiDataHarmonizer`` is the second one.
    """
    def __init__(self, epiconfig : EpiConfig):
        self.epiconfig = epiconfig   

    def _mutate_berlin_districts(self, epidemiology_df : pd.DataFrame) -> pd.DataFrame:
        """
        Mutates all NUTS3 values of the Berlin districts into Berlin key (``'11000'``) 
        for the subsequent aggregation onto nuts3/nuts2/nuts1 levels.
        """

        # change the berlin-districts key to the merged-Berlin key
        epidemiology_df.loc[epidemiology_df['nuts3_key'].isin(berlin_district_ids), 'nuts3_key'] = berlin_id

        return epidemiology_df

    def _add_key_column(self, epi_df : pd.DataFrame, regio_harm : pd.DataFrame) -> pd.DataFrame:
        """adds key column by merging onto harmonization table"""

        regio_harm_cp = regio_harm.copy()

        merge_key = 'nuts3_key'

        # if admininistrative level is nuts3:
        if merge_key != f'{self.epiconfig.level}_key':
            regio_harm_cp = regio_harm_cp.copy()[[merge_key,f'{self.epiconfig.level}_key']].rename(columns = {f'{self.epiconfig.level}_key' : 'key'})

        # if administrative level is nuts1 or nuts2
        else:
            regio_harm_cp       = regio_harm_cp.copy()[[merge_key]]
            regio_harm_cp['key']= regio_harm_cp[merge_key]         

        merge = pd.merge(epi_df, regio_harm_cp , on = merge_key)


        return merge

    def _aggregate_cases(self, epidemiology_df: pd.DataFrame) -> pd.DataFrame:
        """aggregates epidemiology and population data per nuts level"""

        cases_nuts_aggregated = (
            epidemiology_df.groupby([self.epiconfig.temporal_column, 'key'])
            .aggregate({'cases':'sum'}
            ).reset_index())


        return cases_nuts_aggregated

    def _filter_data_on_level(self, df: pd.DataFrame, drop_level: bool = True) -> pd.DataFrame:
        """filter ``'level'`` coluomn"""

        dfc = df.copy()
        dfc = dfc[dfc['level'] == self.epiconfig.level].reset_index(drop = True)

        if drop_level:
            dfc = dfc.drop(columns = ['level'])

        return dfc

    def _set_year_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """create year column based on timestamp"""
        df['year'] = df[self.epiconfig.temporal_column].dt.year 
        return df

    def _merge_epipopdata(self, epidemiology_df : pd.DataFrame, population_df : pd.DataFrame) -> pd.DataFrame:
        """
        merge epidemiology with population data, by extracting the year from epidemiology data. 
        """
        return pd.merge(epidemiology_df, population_df, on = [f'key','year'])

    def _resample(self, epi_df : pd.DataFrame, temporal_freq : Literal['m','w']) -> pd.DataFrame:
        """ 
        resamples timestamp to requested temporal frequency
        """
        # weekly frequency: no change
        if temporal_freq == 'w':
            resampled_df = epi_df

        # monthly frequenty: resample cases on month
        elif temporal_freq == 'm':
            resampled_df = (
                    epi_df.set_index(self.epiconfig.temporal_column)
                    .groupby(f'key')
                    .resample('MS')
                    .agg({
                        'cases':            'sum',
                        'population_size':  'mean',
                    })
                    .reset_index(drop = False)
                )
            
        timestamps: pd.Series[pd.Timestamp] = resampled_df['timestamp']
        resampled_df['year']                = timestamps.dt.year
        resampled_df                        = resampled_df[[self.epiconfig.temporal_column,f'key','cases','year','population_size']]
        return resampled_df

    def _apply_tokenization(self, df: pd.DataFrame, tokenization_map: dict[str, int], drop_key: bool = True) -> pd.DataFrame:
        """
        applies tokeninization found in tokenization map by replacing ``'key'`` column.
        """
        dfc = df.copy()
        dfc[self.epiconfig.id_column] = dfc['key'].map(tokenization_map).astype('Int64')
        dfc.reset_index(drop = True, inplace = True)   

        if drop_key:
            dfc.drop(columns = 'key', inplace=True)

        return dfc  

    def _ensure_geodataframe(self, df: pd.DataFrame | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """turn possible pandas dataframe into geopandas dataframe"""
        df = gpd.GeoDataFrame(df)
        return df

    def _get_keynames(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Region harmonization got columns ``'nutsx_key'`` and ``'nutsx_name'`` for x in [1,2,3] 
        in seperate columns so six columns in total. This function selects the respective key
        and name column.
        """
        cols = [f'{self.epiconfig.level}_key',f'{self.epiconfig.level}_name']
        keynames = df[cols].drop_duplicates().reset_index(drop = True)
        return keynames.rename(columns = {f'{self.epiconfig.level}_key' : 'key'})

    def _return_temporal_summary(self) -> EpiDataTemporalSummary:
        """returns an instance of EpiDataTemporalSummary, based on EpiConfig"""
        return EpiDataTemporalSummary(self.epiconfig.temporal_frequency,
                                    str(self.epiconfig.min_date), 
                                    str(self.epiconfig.max_date),
                                    self.epiconfig.split_trainval,
                                    self.epiconfig.split_valtest,
                                    self.epiconfig.horizon_size,
                                    self.epiconfig.horizon_leadtime,
                                    self.epiconfig.lag_num,
                                    self.epiconfig.sequence_length)        

    def orchestrate(self, rawdata : RawEpiData) -> tuple[HarmonizedEpiData, ContextEpiData]:
        """
        The function that orchestrates all others
        """
        time_start = time.time()

        if self.epiconfig.country == 'germany':
            raw_epidata         = self._mutate_berlin_districts(rawdata.disease)

        else:
            raw_epidata         = rawdata.disease

        epidata             = self._add_key_column(raw_epidata, rawdata.region_harmonization)
        epidata             = self._aggregate_cases(epidata)
        epidata             = self._set_year_col(epidata)

        popdata_lvl         = self._filter_data_on_level(rawdata.population_size)
        local_shapedata     = self._filter_data_on_level(rawdata.shapedata.copy())

        epipopdata          = self._merge_epipopdata(epidata, popdata_lvl)
        epipopdata          = self._resample(epipopdata, self.epiconfig.temporal_frequency)

        keynames            = self._get_keynames(rawdata.region_harmonization.copy())

        # apply tokens
        mapping             = rawdata.tokenization_map
        epipopdata          = self._apply_tokenization(epipopdata, mapping)
        local_shapedata     = self._apply_tokenization(local_shapedata, mapping)
        keynames            = self._apply_tokenization(keynames, mapping, False)

        population_size_data= self._apply_tokenization(popdata_lvl, mapping)

        features = {    
            "population_size_data"      : ("feature_popsize",       "population_size"),              
            "population_density_data"   : ("feature_popdens",       "population_density")
        }

        feature_datasets: dict[str, pd.DataFrame | gpd.GeoDataFrame | None] = {dataname: None for dataname in features}

        for out_name, (feature_flag, raw_attr) in features.items():
            if getattr(self.epiconfig, feature_flag):
                data = getattr(rawdata, raw_attr)
                data = self._filter_data_on_level(data)
                data = self._apply_tokenization(data, mapping)
                feature_datasets[out_name] = data                       


        harmdata = HarmonizedEpiData(
            epidata             = epipopdata,

            population_size    = feature_datasets['population_size_data'],
            population_density = feature_datasets['population_density_data']

        )
        
        ctxdata = ContextEpiData(
            country             = self.epiconfig.country,
            level               = self.epiconfig.level,
            global_shapedata    = self._ensure_geodataframe(rawdata.shapedata),
            local_shapedata     = self._ensure_geodataframe(local_shapedata),
            population_size     = population_size_data,
            nodenames           = keynames,
            region_harmonization= rawdata.region_harmonization,
            tokenization_map    = mapping,
            temporal_summary    = self._return_temporal_summary()            
        )

        time_end = time.time()
        time_elapsed = time_end - time_start
        return harmdata, ctxdata
