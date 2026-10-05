import pandas as pd 
import numpy as np

from ...dataloading.databuilders import BaseLineDataBuilder 
from .baselinemodel import BaseLineModel 

from ...utils import DataSetSplit

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

        self._temporal_idx_column = 'tidx'

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        assert isinstance(self.databuilder, BaseLineDataBuilder)
        id_col, t_col, idx_col = (self.epiconfig.id_column,
                                self.epiconfig.temporal_column,
                                self._temporal_idx_column)

        df = (self.databuilder.dataloader_main
                .sort_values([id_col, t_col]).copy())
        df[idx_col] = self._get_seasonal_index(df)

        # node-specific seasonal mean, fit on train only
        seasonal_mean = (df[df['train']]
                        .groupby([id_col, idx_col])['target'].mean()
                        .rename('pred').reset_index())

        # point prediction for ALL rows (train, val, test)
        df = df.merge(seasonal_mean, on=[id_col, idx_col])

        if self.epiconfig._prediction_mode == 'interval':
            assert self.epiconfig.quantiles is not None
            table = self._compute_residual_quantiles(['train', 'val'], df)
            for i, q in enumerate(self.epiconfig.quantiles):
                offset = df[idx_col].map(table[q])
                df[f'pred_q{i+1}'] = (df['pred'] + offset).clip(lower=0)

        df = df[df[dataset]]
        out = df[[id_col, t_col, 'target'] + self.prediction_columns]

        for hh in range(self.databuilder.dataorchestrator.config.horizon_size):
            self.predictions.add_horizon_predictions(dataset, self._transform(out), hh)

        self._update_status('forecasted')

    def _compute_residual_quantiles(self,
                                    dataset: DataSetSplit | list[DataSetSplit],
                                    df: pd.DataFrame) -> pd.DataFrame:
        quantiles  = np.array(self.epiconfig.quantiles)
        split_cols = [dataset] if isinstance(dataset, str) else dataset

        d = df[df[split_cols].any(axis=1)].dropna(subset=['pred', 'target'])
        resid = d['target'] - d['pred']
        return resid.groupby(d[self._temporal_idx_column]).quantile(quantiles).unstack()

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
        