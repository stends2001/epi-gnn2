"""
Split-conformal style residual-quantile intervals.

Shared by the baseline models (``Persistence``, ``SeasonalAverage``) and by the
conformal wrapper around point-forecasting GNNs. The idea is the same everywhere:

1. compute residuals ``target - pred`` on a calibration pool (never the test split),
2. take empirical quantiles of those residuals per group (e.g. week of year),
3. add the quantile offsets to the point forecast.

Two residual scales are supported:

- ``'additive'``: residuals on the original scale. Intervals have the same absolute
  width for every node in a group, which is too wide for small regions and too
  narrow for large ones when nodes differ in size.
- ``'log1p'``: residuals of ``log1p(target) - log1p(pred)``. This is a relative error,
  so one table serves nodes of different size, the back-transformed interval can
  never go negative, and it does not collapse to zero width when ``pred == 0``.
  Requires non-negative predictions and targets.

Thin bins (few calibration residuals in a group) give noisy quantiles. Groups with
fewer than ``min_obs`` residuals fall back to the quantiles pooled over all groups.
Groups that never appear in the calibration pool (e.g. ISO week 53) also use the
pooled quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

ResidualScale = Literal['additive', 'log1p']


@dataclass
class ResidualQuantileTable:
    """
    Residual quantiles per group, with the pooled fallback and bin sizes.

    Attributes
    ----------
    table : pd.DataFrame
        Index: group key. Columns: quantile levels. Thin bins already replaced by
        the pooled quantiles.
    pooled : pd.Series
        Quantiles over all calibration residuals, indexed by quantile level.
    n_obs : pd.Series
        Number of calibration residuals per group (before fallback). Inspect this
        to decide whether the grouping is too fine.
    scale : ResidualScale
        Scale on which the residuals were computed.
    min_obs : int
        Threshold below which a group falls back to ``pooled``.
    """
    table:   pd.DataFrame
    pooled:  pd.Series
    n_obs:   pd.Series
    scale:   ResidualScale
    min_obs: int
    quantiles: list[float] = field(default_factory=list)

    @property
    def fallback_groups(self) -> list:
        """Groups whose quantiles were replaced by the pooled quantiles."""
        return self.n_obs[self.n_obs < self.min_obs].index.tolist()

    def offsets_for(self, group: pd.Series) -> pd.DataFrame:
        """
        Offsets per row for the given group keys. Unknown groups get the pooled
        quantiles. Columns are quantile levels.
        """
        out = pd.DataFrame(index=group.index)
        for q in self.quantiles:
            out[q] = group.map(self.table[q]).fillna(self.pooled[q]).astype(float)
        return out

    def apply(self, pred: pd.Series, group: pd.Series, clip_lower: float | None = 0.0) -> pd.DataFrame:
        """
        Build quantile forecasts from point forecasts.

        Returns a DataFrame with one column per quantile level, index aligned
        with ``pred``.
        """
        offsets = self.offsets_for(group)
        out     = pd.DataFrame(index=pred.index)

        for q in self.quantiles:
            if self.scale == 'additive':
                values = pred + offsets[q]
            elif self.scale == 'log1p':
                values = np.expm1(np.log1p(pred.clip(lower=0)) + offsets[q])
            else:
                raise ValueError(f'unknown residual scale {self.scale!r}')

            out[q] = values.clip(lower=clip_lower) if clip_lower is not None else values

        return out


def residuals(target: pd.Series, pred: pd.Series, scale: ResidualScale) -> pd.Series:
    """Residuals ``target - pred`` on the requested scale."""
    if scale == 'additive':
        return target - pred
    if scale == 'log1p':
        if (target < 0).any() or (pred < 0).any():
            raise ValueError("log1p residuals need non-negative target and pred. "
                             "Use residual_scale='additive' for transformed data.")
        return np.log1p(target) - np.log1p(pred)
    raise ValueError(f'unknown residual scale {scale!r}')


def residual_quantile_table(target:    pd.Series,
                            pred:      pd.Series,
                            group:     pd.Series,
                            quantiles: list[float],
                            scale:     ResidualScale = 'additive',
                            min_obs:   int = 30) -> ResidualQuantileTable:
    """
    Empirical residual quantiles per group, with pooled fallback for thin bins.

    Parameters
    ----------
    target, pred, group : pd.Series
        Aligned series over the calibration pool. Rows with NaN in ``target`` or
        ``pred`` are dropped.
    quantiles : list[float]
        Quantile levels in (0, 1).
    scale : {'additive', 'log1p'}
        Residual scale, see module docstring.
    min_obs : int
        Groups with fewer residuals than this use the pooled quantiles.
    """
    mask  = target.notna() & pred.notna()
    res   = residuals(target[mask], pred[mask], scale)
    grp   = group[mask]
    q_arr = np.asarray(quantiles, dtype=float)

    if len(res) == 0:
        raise ValueError('No calibration residuals: the calibration pool is empty.')

    pooled = res.quantile(q_arr)
    pooled.index = list(quantiles)

    table = res.groupby(grp).quantile(q_arr).unstack()
    table.columns = list(quantiles)

    n_obs = res.groupby(grp).size().rename('n_obs')

    thin = n_obs[n_obs < min_obs].index
    if len(thin) > 0:
        for q in quantiles:
            table.loc[thin, q] = pooled[q]

    return ResidualQuantileTable(table     = table,
                                 pooled    = pooled,
                                 n_obs     = n_obs,
                                 scale     = scale,
                                 min_obs   = min_obs,
                                 quantiles = list(quantiles))
