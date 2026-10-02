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
    
    def _compute_residual_quantiles(self, dataset: DataSetSplit | list[DataSetSplit]) -> dict[int, pd.DataFrame]:
        """
        Per-horizon residual quantiles based on column 'pred', keyed by horizon.
        Each value is a DataFrame indexed by seasonal index, columned by quantile.

        ``dataset`` may be a single split or a list of splits (e.g.
        ``['train', 'val']``) to pool residuals across more than one —
        legitimate for Persistence specifically, since it has no fitted
        parameters and therefore no optimism-bias risk from including
        train in its own calibration pool.
        """
        quantiles        = self.databuilder.dataorchestrator.config.quantiles
        horizon_leadtime = self.databuilder.dataorchestrator.config.horizon_leadtime
        horizon_size     = self.databuilder.dataorchestrator.config.horizon_size

        split_cols = [dataset] if isinstance(dataset, str) else dataset

        tables: dict[int, pd.DataFrame] = {}

        for hh in range(horizon_size):
            timeshift_num = int(hh + horizon_leadtime)

            df = self.databuilder.dataloader_main
            df = df.sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column]).copy()

            df['pred'] = df.groupby(self.epiconfig.id_column)['target'].shift(timeshift_num)

            # filter to the pooled split(s) AFTER shifting
            df = df[df[split_cols].any(axis=1)].dropna(subset=['pred', 'target'])

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
            self._residual_quantiles = self._compute_residual_quantiles(['train','val'])

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