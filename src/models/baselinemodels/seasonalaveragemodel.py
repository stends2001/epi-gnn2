from typing import Literal

import pandas as pd
import numpy as np

from ...dataloading.databuilders import BaseLineDataBuilder
from .baselinemodel import BaseLineModel
from ..utils.conformal import ResidualScale, residual_quantile_table

from ...utils import DataSetSplit

TrainResiduals = Literal['in_sample', 'leave_one_year_out']

class SeasonalAverage(BaseLineModel):
    """
    Seasonal Average model returns the node-specific average per seasonal index
    (week of year for weekly data), fit on the training split.

    The forecast does not depend on the horizon, so one residual table serves all
    horizons in interval mode.

    Parameters
    ----------
    train_residuals : {'in_sample', 'leave_one_year_out'}
        How train-split residuals enter the calibration pool. The seasonal mean is
        fit on train, so ``'in_sample'`` residuals are too small (with 5 years per
        bin the residual SD is about 11% too narrow). ``'leave_one_year_out'``
        predicts each train row from the other train years of the same node and
        bin, which gives honest residuals. Val residuals always use the full
        train mean.

    Other parameters are documented in ``BaseLineModel``.

    See Also
    --------
    ``BaseModel``
        Parent class of all models.
    ``BaseLineModel``
        Parent class of baseline models.
    """
    def __init__(self,
                 databuilder : BaseLineDataBuilder,
                 name: str = 'seasonal_average_model',
                 residual_scale: ResidualScale = 'additive',
                 min_bin_obs: int = 30,
                 calibration_splits: list[Literal['train', 'val']] | None = None,
                 train_residuals: TrainResiduals = 'leave_one_year_out'):

        super().__init__(databuilder, name,
                         residual_scale     = residual_scale,
                         min_bin_obs        = min_bin_obs,
                         calibration_splits = calibration_splits)

        self._temporal_idx_column = 'tidx'
        self._year_column         = '_year'
        self.train_residuals      = train_residuals
        self.config_info['train_residuals'] = train_residuals

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        assert isinstance(self.databuilder, BaseLineDataBuilder)
        id_col, t_col, idx_col = (self.epiconfig.id_column,
                                  self.epiconfig.temporal_column,
                                  self._temporal_idx_column)

        df = (self.databuilder.dataloader_main
                .sort_values([id_col, t_col]).copy())
        df[idx_col] = self._get_seasonal_index(df)
        df[self._year_column] = self._get_year(df)

        df['pred'] = self._seasonal_mean_prediction(df)

        if self.epiconfig._prediction_mode == 'interval':
            assert self.epiconfig.quantiles is not None

            calib_pred = df['pred'].copy()
            if self.train_residuals == 'leave_one_year_out':
                train_mask = df['train']
                calib_pred[train_mask] = self._leave_one_year_out_prediction(df)[train_mask]

            pool  = df[self.calibration_splits].any(axis=1)
            table = residual_quantile_table(
                target    = df.loc[pool, 'target'],
                pred      = calib_pred[pool],
                group     = df.loc[pool, idx_col],
                quantiles = self.epiconfig.quantiles,
                scale     = self.residual_scale,
                min_obs   = self.min_bin_obs,
            )

            quantile_preds = table.apply(df['pred'], df[idx_col], clip_lower=0.0)
            for i, q in enumerate(self.epiconfig.quantiles):
                df[f'pred_q{i+1}'] = quantile_preds[q]

        df  = df[df[dataset]]
        out = df[[id_col, t_col, 'target'] + self.prediction_columns]

        for hh in range(self.databuilder.dataorchestrator.config.horizon_size):
            if self.epiconfig._prediction_mode == 'interval':
                # same table for every horizon; stored per horizon for calibration_summary()
                self.calibration_tables[hh] = table
            self.predictions.add_horizon_predictions(dataset, self._transform(out), hh)

        self._update_status('forecasted')

    # ======= HIDDEN METHODS ======= #
    def _get_year(self, df: pd.DataFrame) -> pd.Series:
        """Year that matches the seasonal index (ISO year for weekly data)."""
        if self.databuilder.dataorchestrator.config.temporal_frequency == 'w':
            return df[self.epiconfig.temporal_column].dt.isocalendar().year.astype(int)
        return df[self.epiconfig.temporal_column].dt.year.astype(int)

    def _seasonal_mean_prediction(self, df: pd.DataFrame) -> pd.Series:
        """
        Node-specific train mean per seasonal index, for every row.

        Bins that never occur in train for a node (typically ISO week 53) use the
        previous bin of that node, and otherwise the node's overall train mean.
        A left merge keeps every row, so the prediction manager receives a complete
        time range.
        """
        id_col, idx_col = self.epiconfig.id_column, self._temporal_idx_column
        train = df[df['train']]

        seasonal_mean = train.groupby([id_col, idx_col])['target'].mean()
        node_mean     = train.groupby(id_col)['target'].mean()

        keys = pd.MultiIndex.from_arrays([df[id_col], df[idx_col]])
        pred = pd.Series(seasonal_mean.reindex(keys).to_numpy(), index=df.index)

        missing = pred.isna()
        if missing.any():
            max_idx  = int(df[idx_col].max())
            prev_idx = (df.loc[missing, idx_col] - 2) % max_idx + 1     # 1 -> max_idx
            prev_keys = pd.MultiIndex.from_arrays([df.loc[missing, id_col], prev_idx])
            pred[missing] = seasonal_mean.reindex(prev_keys).to_numpy()

            still = pred.isna()
            pred[still] = df.loc[still, id_col].map(node_mean)

        return pred

    def _leave_one_year_out_prediction(self, df: pd.DataFrame) -> pd.Series:
        """
        For train rows: mean of the same node and bin over the OTHER train years.
        NaN when a bin only has one train year.
        """
        id_col, idx_col, yr_col = self.epiconfig.id_column, self._temporal_idx_column, self._year_column
        train = df[df['train']]

        grp      = train.groupby([id_col, idx_col])['target']
        total    = grp.transform('sum')
        count    = grp.transform('count')

        grp_y    = train.groupby([id_col, idx_col, yr_col])['target']
        total_y  = grp_y.transform('sum')
        count_y  = grp_y.transform('count')

        denom = (count - count_y).replace(0, np.nan)
        loo   = (total - total_y) / denom

        return loo.reindex(df.index)
