import pandas as pd 
import numpy as np

from ...dataloading.databuilders import BaseLineDataBuilder 
from .baselinemodel import BaseLineModel 

from ...utils import DataSetSplit

class Persistence(BaseLineModel):
    """ 
    Persistence model returns the most recent observation as prediction.

    See Also
    --------
    ``BaseModel``
        Parent class of all models.
    ``BaseLineModel``
        Parent class of baseline models.
    """
    def __init__(self, 
                 databuilder : BaseLineDataBuilder,                 
                 name: str = 'persistence_model'):
        
        super().__init__(databuilder, name)

    def _get_seasonal_index(self, df: pd.DataFrame) -> pd.Series:
        """Returns seasonal index series based on temporal frequency"""
        freq = self.databuilder.dataorchestrator.config.temporal_frequency
        if freq == 'w':
            return df[self.epiconfig.temporal_column].dt.isocalendar().week.astype(int)
        elif freq == 'd':
            return df[self.epiconfig.temporal_column].dt.dayofyear.astype(int)
        elif freq == 'm':
            return df[self.epiconfig.temporal_column].dt.month
        else:
            raise ValueError(f'Invalid temporal frequency found for ClimaScale model: {freq}')

    def _compute_residual_quantiles(self, dataset: str) -> dict[int, pd.DataFrame]:
        """
        Per-horizon residual quantiles based on column 'pred', keyed by horizon.
        Each value is a DataFrame indexed by seasonal index, columned by quantile.
        """
        quantiles       = self.databuilder.dataorchestrator.config.quantiles
        horizon_leadtime = self.databuilder.dataorchestrator.config.horizon_leadtime
        horizon_size    = self.databuilder.dataorchestrator.config.horizon_size

        tables: dict[int, pd.DataFrame] = {}

        for hh in range(horizon_size):
            timeshift_num = int(hh + horizon_leadtime)

            # shift on the FULL series first (matches forecast()'s order),
            # so val's boundary-adjacent rows still get a valid shifted value
            df = self.databuilder.dataloader_main
            df = df.sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column]).copy()

            df['pred'] = df.groupby(self.epiconfig.id_column)['target'].shift(timeshift_num)

            # filter to split AFTER shifting
            df = df[df[dataset]].dropna(subset=['pred', 'target'])

            residuals = df['target'] - df['pred']
            t_idx     = self._get_seasonal_index(df)

            tables[hh] = (
                residuals.groupby(t_idx)
                          .quantile(np.array(quantiles))
                          .unstack()
            )

        return tables

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        """
        Forecast for set dataset
        """
        assert isinstance(self.databuilder, BaseLineDataBuilder)

        if self.epiconfig._prediction_mode == 'interval':
            self._residual_quantiles = self._compute_residual_quantiles('val')

        for hh in range(self.databuilder.dataorchestrator.config.horizon_size):

            # get shift between target and pred
            timeshift_num = int(hh + self.databuilder.dataorchestrator.config.horizon_leadtime)
            evaluation_df = self.databuilder.dataloader_main

            evaluation_df = evaluation_df.sort_values([
                self.epiconfig.id_column, 
                self.epiconfig.temporal_column]
            ).copy()

            # group by, and shift 'target' by ``timeshift_num``.
            # that is the prediction: the shifted 'target'.
            persistence_pred = evaluation_df.groupby(
                self.epiconfig.id_column
                )['target'].shift(timeshift_num)

            # point-predictions:
            evaluation_df['pred'] = persistence_pred

            if self.epiconfig._prediction_mode == 'interval':
                assert self.epiconfig.quantiles != None
                assert self.epiconfig._num_quantiles != None

                t_idx = self._get_seasonal_index(evaluation_df)
                horizon_table = self._residual_quantiles[hh]   # this horizon's table

                for i, q in enumerate(self.epiconfig.quantiles):
                    offset = t_idx.map(horizon_table[q])
                    evaluation_df[f'pred_q{i+1}'] = (persistence_pred + offset).clip(lower=0)

           # filter on dataset train/val/test
            evaluation_df = evaluation_df[evaluation_df[dataset]]
            evaluation_dataset = evaluation_df[
                [self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + self.prediction_columns
                ]                
        
            self.predictions.add_horizon_predictions(dataset, self._transform(evaluation_dataset), hh)

        self._update_status('forecasted')   