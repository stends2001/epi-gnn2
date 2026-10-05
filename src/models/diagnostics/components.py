"""
Tables of the endemic / epidemic / neighbourhood decomposition of ``HHH4Model``.

Shares are reported in two ways:

- **mu-weighted** (``share_*``): sum of a component / sum of mu over the rows. This
  answers "which fraction of the expected cases comes from each branch" and is not
  dominated by off-season weeks, where mu is tiny and per-row shares are noise.
- **row median** (``median_share_*``): the typical per-row share, as in
  ``forecast_components().describe()``.

Rows are split into in-season and off-season by the TARGET week (t0 + lead + h),
with the season window from ``intervalmetrics.DEFAULT_SEASONS``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.intervalmetrics import season_mask, season_weeks_for

COMPONENTS = ('endemic', 'epidemic', 'neighbourhood')


def components_with_target_time(model, dataset: str = 'test') -> pd.DataFrame:
    """``model.forecast_components`` plus a ``target_time`` column."""
    comp = model.forecast_components(dataset)
    t_col = model.epiconfig.temporal_column
    steps = model.epiconfig.horizon_leadtime + comp['horizon']
    if model.epiconfig.temporal_frequency == 'w':
        comp['target_time'] = pd.to_datetime(comp[t_col]) + pd.to_timedelta(7 * steps, unit='D')
    else:
        comp['target_time'] = [pd.Timestamp(t) + pd.DateOffset(months=int(s))
                               for t, s in zip(comp[t_col], steps)]
    return comp


def _summarise(g: pd.DataFrame) -> dict:
    mu_sum = g['mu'].sum()
    row = {
        'n_rows':      len(g),
        'mean_mu':     g['mu'].mean(),
        'mean_target': g['target'].mean(),
        'bias_ratio':  mu_sum / max(g['target'].sum(), 1e-12),
    }
    for c in COMPONENTS:
        row[f'share_{c}']        = g[c].sum() / max(mu_sum, 1e-12)
    for c in COMPONENTS:
        row[f'median_share_{c}'] = g[f'share_{c}'].median()
    row['frac_rows_neighbourhood_gt_10pct'] = float((g['share_neighbourhood'] > 0.10).mean())
    row['alpha_mean'] = g['alpha'].mean()
    return row


def component_table(model,
                    dataset: str = 'test',
                    season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    Component shares for all weeks, in-season and off-season, per horizon.

    Columns: period, horizon, n_rows, mean_mu, mean_target, bias_ratio (sum mu /
    sum target), share_* (mu-weighted), median_share_*, the fraction of rows where
    the neighbourhood carries more than 10% of mu, and the mean dispersion.
    """
    comp  = components_with_target_time(model, dataset)
    weeks = season_weeks or season_weeks_for(model.epiconfig.disease)
    comp['in_season'] = season_mask(comp['target_time'], *weeks).to_numpy()

    rows = []
    for hh, g in comp.groupby('horizon'):
        for period, sub in [('all', g), ('in-season', g[g['in_season']]), ('off-season', g[~g['in_season']])]:
            if len(sub) == 0:
                continue
            rows.append({'period': period, 'horizon': hh, **_summarise(sub)})
    out = pd.DataFrame(rows)
    out.insert(0, 'model', model.name)
    return out


def component_by_node(model,
                      dataset: str = 'test',
                      season: str | None = 'in',
                      season_weeks: tuple[int, int] | None = None,
                      horizon: int = 0) -> pd.DataFrame:
    """
    Mu-weighted component shares per node (default: in-season), with node names
    and the learned node parameters, ready for maps or sorting.
    """
    comp  = components_with_target_time(model, dataset)
    comp  = comp[comp['horizon'] == horizon]
    if season is not None:
        weeks = season_weeks or season_weeks_for(model.epiconfig.disease)
        mask  = season_mask(comp['target_time'], *weeks).to_numpy()
        comp  = comp[mask if season == 'in' else ~mask]

    id_col = model.epiconfig.id_column
    rows = [{id_col: node, **_summarise(g)} for node, g in comp.groupby(id_col)]
    out  = pd.DataFrame(rows)

    params = model.node_parameters()
    params = params[params['horizon'] == horizon].drop(columns='horizon')
    out = out.merge(params, on=id_col, how='left')
    return out


def compare_components(models,
                       dataset: str = 'test',
                       period: str = 'in-season',
                       season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    Side-by-side component shares of several HHH4 models (diseases, graphs, seeds).

    ``models`` is a list or a dict {label: model}. Each model uses its own disease
    season unless ``season_weeks`` is given.
    """
    items = models.items() if isinstance(models, dict) else [(m.name, m) for m in models]
    parts = []
    for label, m in items:
        t = component_table(m, dataset, season_weeks)
        t = t[t['period'] == period].copy()
        t['model'] = label
        parts.append(t)
    cols = ['model', 'horizon', 'n_rows', 'bias_ratio',
            'share_endemic', 'share_epidemic', 'share_neighbourhood',
            'median_share_neighbourhood', 'frac_rows_neighbourhood_gt_10pct', 'alpha_mean']
    return pd.concat(parts, ignore_index=True)[cols]
