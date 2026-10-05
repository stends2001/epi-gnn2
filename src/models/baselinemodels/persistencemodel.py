from typing import Literal

import pandas as pd

from ...dataloading.databuilders import BaseLineDataBuilder
from .baselinemodel import BaseLineModel
from ..utils.conformal import ResidualScale, residual_quantile_table

from ...utils import DataSetSplit

class Persistence(BaseLineModel):
    """
    Persistence model returns the most recent observation as prediction.

    In interval mode, quantiles come from residuals of the same persistence
    forecast on the calibration pool (default train + val), grouped per
    (horizon, seasonal index). Persistence has no fitted parameters, so its
    train residuals are honest out-of-sample errors.

    See Also
    --------
    ``BaseModel``
        Parent class of all models.
    ``BaseLineModel``
        Parent class of baseline models, documents the calibration options.
    """
    def __init__(self,
                 databuilder : BaseLineDataBuilder,
                 name: str = 'persistence_model',
                 residual_scale: ResidualScale = 'additive',
                 min_bin_obs: int = 30,
                 calibration_splits: list[Literal['train', 'val']] | None = None):

        super().__init__(databuilder, name,
                         residual_scale     = residual_scale,
                         min_bin_obs        = min_bin_obs,
                         calibration_splits = calibration_splits)

    def _persistence_frame(self, hh: int) -> pd.DataFrame:
        """Main data with the persistence point forecast for horizon ``hh`` in 'pred'."""
        timeshift_num = int(hh + self.databuilder.dataorchestrator.config.horizon_leadtime)

        df = self.databuilder.dataloader_main
        df = df.sort_values([self.epiconfig.id_column, self.epiconfig.temporal_column]).copy()

        # the prediction is the target, shifted by ``timeshift_num`` within each node
        df['pred'] = df.groupby(self.epiconfig.id_column)['target'].shift(timeshift_num)
        return df

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        """
        Forecast for set dataset
        """
        assert isinstance(self.databuilder, BaseLineDataBuilder)
        interval_mode = self.epiconfig._prediction_mode == 'interval'

        for hh in range(self.databuilder.dataorchestrator.config.horizon_size):

            evaluation_df = self._persistence_frame(hh)

            if interval_mode:
                assert self.epiconfig.quantiles is not None
                t_idx = self._get_seasonal_index(evaluation_df)

                # calibration pool: filter AFTER shifting, so the first rows of a split
                # still use the last observations of the previous split as forecast
                pool = evaluation_df[self.calibration_splits].any(axis=1)

                table = residual_quantile_table(
                    target    = evaluation_df.loc[pool, 'target'],
                    pred      = evaluation_df.loc[pool, 'pred'],
                    group     = t_idx[pool],
                    quantiles = self.epiconfig.quantiles,
                    scale     = self.residual_scale,
                    min_obs   = self.min_bin_obs,
                )
                self.calibration_tables[hh] = table

                quantile_preds = table.apply(evaluation_df['pred'], t_idx, clip_lower=0.0)
                for i, q in enumerate(self.epiconfig.quantiles):
                    evaluation_df[f'pred_q{i+1}'] = quantile_preds[q]

            # filter on dataset train/val/test
            evaluation_df = evaluation_df[evaluation_df[dataset]]
            evaluation_dataset = evaluation_df[
                [self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + self.prediction_columns
                ]

            self.predictions.add_horizon_predictions(dataset, self._transform(evaluation_dataset), hh)

        self._update_status('forecasted')
