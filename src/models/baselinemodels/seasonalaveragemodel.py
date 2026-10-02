import pandas as pd 

from ...dataloading.databuilders import BaseLineDataBuilder 
from ...utils import DataSetSplit

from .baselinemodel import BaseLineModel 

class SeasonalAverage(BaseLineModel):
    """ 
    Seasonal Average model returns the average of the week index based on training data.

    See Also
    --------
    ``BaseModel``
        Parent class of all models.
    ``BaseLineModel``
        Parent class of baseline models.
    """
    def __init__(self, 
                 databuilder : BaseLineDataBuilder,                 
                 name: str = 'seasonal_average_model'):
        
        super().__init__(databuilder, name)

        self.seasonal_averages = self._get_temporal_averages(databuilder.dataloader_main)

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        """
        Forecast for set dataset
        """
        for hh in range(self.databuilder.dataorchestrator.config.horizon_size):

            dl              = self.databuilder.dataloader_main

            if not isinstance(dl, pd.DataFrame):
                raise ValueError()

            # filter on dataset
            evaluation_df = dl[dl[dataset]]
            evaluation_df = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target']]

            # get seasonal averages: average per week idx 
            evaluation_df = self._get_seasonal_indexes(evaluation_df)
            evaluation_df = pd.merge(evaluation_df, self.seasonal_averages, on=[self.epiconfig.id_column, 't_idx'])

            evaluation_df = evaluation_df.rename(columns={'seasonal_mean': 'pred'}).drop(columns=['t_idx'])

            evaluation_dataset = evaluation_df[[self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + [self.pred_col]]
            self.predictions.add_horizon_predictions(dataset, self._transform(evaluation_dataset), hh)            
           
        self._update_status('forecasted')   
    
    def _get_seasonal_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds t_idx column based on temporal frequency"""
        
        dfc         = df.copy()
        freq        = self.databuilder.dataorchestrator.config.temporal_frequency

        timestamp: pd.Series[pd.Timestamp]  = dfc[self.epiconfig.temporal_column]   

        # add time-index column 't_idx
        if freq == 'w':
            dfc['t_idx'] = timestamp.dt.isocalendar().week.astype(int)
        elif freq == 'm':
            dfc['t_idx'] = timestamp.dt.month
        else:
            raise ValueError(f'Invalid temporal frequency found for ClimaScale model: {freq}')
        
        return dfc
    
    def _get_temporal_averages(self, dataloader_main: pd.DataFrame) -> pd.DataFrame:
        """Returns a df with the average target per (node, seasonal timepoint) over training data"""
        dataloader_train = dataloader_main[dataloader_main['train']]
        seasonal_index   = self._get_seasonal_indexes(dataloader_train)       

        return (seasonal_index.groupby([self.epiconfig.id_column, 't_idx'])['target']
                    .mean()
                    .reset_index()
                    .rename(columns={'target': 'seasonal_mean'}))