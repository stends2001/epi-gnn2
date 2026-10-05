"""
Interval / quantile forecast metrics.

All functions take a prediction frame with the columns ``target`` and
``pred_q1 ... pred_qN`` (as stored by ``PredictionManager``) and the matching,
validated quantile levels (odd, sorted, symmetric around 0.5).

Metrics
-------
- WIS (weighted interval score; Bracher et al. 2021, PLoS Comput Biol):
      WIS = 1 / (K + 1/2) * ( 1/2 * |y - median| + sum_k alpha_k / 2 * IS_alpha_k )
  with K central intervals at alpha_k = 2 * q_k and the interval score
      IS_alpha = (u - l) + 2/alpha * (l - y) * 1[y < l] + 2/alpha * (y - u) * 1[y > u].
  WIS equals 1 / (K + 1/2) times the summed pinball loss over all 2K + 1
  levels, and approximates the CRPS when there are many levels.
- Empirical coverage and mean width per central interval (nominal 1 - 2 q_k).
- PIT-style quantile ranks: the fraction of quantile levels below the target.
  For a calibrated forecast, the ranks are roughly uniform.
- Count forecasts: plain interval coverage is biased upwards when quantiles are
  whole numbers (a 50% interval [1, 3] around a mean of 2 holds ~70%). For models
  with an NB predictive distribution use the randomised PIT
  (``nb_randomized_pit``, ``pit_coverage``), which is exactly uniform when the
  forecast is calibrated.

Report these per horizon and per node group, not only pooled: pooled coverage can
look fine while large and small regions are mis-covered in opposite directions.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _quantile_matrix(df: pd.DataFrame, quantiles: list[float]) -> np.ndarray:
    cols = [f'pred_q{i+1}' for i in range(len(quantiles))]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f'prediction frame is missing quantile columns {missing}')
    return df[cols].to_numpy(dtype=float)


def _check_quantiles(quantiles: list[float]) -> None:
    q = np.asarray(quantiles, dtype=float)
    mid = len(q) // 2
    if len(q) % 2 == 0 or abs(q[mid] - 0.5) > 1e-9:
        raise ValueError('quantiles must be odd-length with 0.5 in the middle')
    if not np.allclose(q + q[::-1], 1.0):
        raise ValueError('quantiles must be symmetric around 0.5')


def interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Interval score of the central (1 - alpha) interval, per row."""
    return ((upper - lower)
            + 2.0 / alpha * np.clip(lower - y, 0, None)
            + 2.0 / alpha * np.clip(y - upper, 0, None))


def wis(df: pd.DataFrame, quantiles: list[float], decompose: bool = False) -> pd.Series | pd.DataFrame:
    """
    Weighted interval score per row.

    With ``decompose=True`` returns a frame with ``wis`` and its three parts:
    ``dispersion`` (width), ``underprediction`` and ``overprediction``
    (penalties for y above / below the intervals), which sum to ``wis``.
    """
    _check_quantiles(quantiles)
    Q   = _quantile_matrix(df, quantiles)
    y   = df['target'].to_numpy(dtype=float)
    K   = len(quantiles) // 2
    med = Q[:, K]

    disp  = np.zeros_like(y)
    under = 0.5 * np.clip(y - med, 0, None)     # median term split by side
    over  = 0.5 * np.clip(med - y, 0, None)

    for k in range(K):
        alpha = 2.0 * quantiles[k]
        l, u  = Q[:, k], Q[:, -1 - k]
        w     = alpha / 2.0
        disp  += w * (u - l)
        under += w * 2.0 / alpha * np.clip(y - u, 0, None)
        over  += w * 2.0 / alpha * np.clip(l - y, 0, None)

    norm  = 1.0 / (K + 0.5)
    total = norm * (disp + under + over)

    if not decompose:
        return pd.Series(total, index=df.index, name='wis')
    return pd.DataFrame({'wis'            : total,
                         'dispersion'     : norm * disp,
                         'underprediction': norm * under,
                         'overprediction' : norm * over}, index=df.index)


def coverage_and_width(df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """
    Empirical coverage and mean width per central interval.

    Returns one row per interval with ``nominal``, ``coverage``, ``width``,
    ``below`` (share of y under the lower bound) and ``above``.
    """
    _check_quantiles(quantiles)
    Q = _quantile_matrix(df, quantiles)
    y = df['target'].to_numpy(dtype=float)
    K = len(quantiles) // 2

    rows = []
    for k in range(K):
        l, u = Q[:, k], Q[:, -1 - k]
        rows.append({
            'lower_q' : quantiles[k],
            'upper_q' : quantiles[-1 - k],
            'nominal' : 1.0 - 2.0 * quantiles[k],
            'coverage': float(np.mean((y >= l) & (y <= u))),
            'below'   : float(np.mean(y < l)),
            'above'   : float(np.mean(y > u)),
            'width'   : float(np.mean(u - l)),
        })
    return pd.DataFrame(rows)


def nb_randomized_pit(y, mu, alpha, seed: int = 0) -> np.ndarray:
    """
    Randomised PIT for NB2(mu, alpha) count forecasts (Czado, Gneiting & Held 2009).

    For a count y the CDF jumps at y, so the PIT is drawn uniformly between
    F(y - 1) and F(y). For a calibrated count forecast the result is exactly
    uniform, which the plain interval coverage of a count forecast is not:
    with whole-number quantiles a "50% interval" holds more than 50% of the
    probability, most of all for small counts.
    """
    from scipy.stats import nbinom
    y     = np.asarray(y, dtype=float)
    mu    = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
    alpha = np.clip(np.asarray(alpha, dtype=float), 1e-10, None)
    r     = 1.0 / alpha
    p     = r / (r + mu)
    yi    = np.round(y)
    hi    = nbinom.cdf(yi, r, p)
    lo    = np.where(yi > 0, nbinom.cdf(yi - 1, r, p), 0.0)
    u     = np.random.default_rng(seed).uniform(size=y.shape)
    return lo + u * (hi - lo)


def pit_coverage(pit, quantiles: list[float]) -> pd.DataFrame:
    """
    Count-aware coverage from (randomised) PIT values: the share of PIT values
    inside the central interval [q, 1 - q]. Expected value: 1 - 2q exactly, also
    for small counts. Same layout as ``coverage_and_width`` (without widths).
    """
    _check_quantiles(quantiles)
    pit = np.asarray(pit, dtype=float)
    K = len(quantiles) // 2
    rows = []
    for k in range(K):
        q = quantiles[k]
        rows.append({'lower_q': q, 'upper_q': quantiles[-1 - k], 'nominal': 1.0 - 2.0 * q,
                     'coverage': float(np.mean((pit >= q) & (pit <= 1 - q))),
                     'below': float(np.mean(pit < q)), 'above': float(np.mean(pit > 1 - q))})
    return pd.DataFrame(rows)


def model_pit(model, dataset: str = 'test', season: str | None = None,
              season_weeks: tuple[int, int] | None = None) -> np.ndarray | None:
    """
    Randomised PIT values of a model with a full predictive distribution: NB
    (``HHH4Model``) or simulated (the R hhh4 wrapper, via ``pit_bounds``),
    filtered by season on the target week.
    None for models without a full predictive distribution (baselines, quantile heads).
    """
    if hasattr(model, 'predictive_nb'):
        df = model.predictive_nb(dataset)                # columns: target_time, node, target, mu, alpha
    elif hasattr(model, 'pit_bounds'):
        df = model.pit_bounds(dataset)                   # columns: target_time, lo = F(y-1), hi = F(y)
    else:
        return None
    if season is not None:
        weeks = season_weeks or season_weeks_for(getattr(model.epiconfig, 'disease', None))
        m = season_mask(df['target_time'], *weeks).to_numpy()
        df = df[m if season == 'in' else ~m]
    if 'mu' in df.columns:
        return nb_randomized_pit(df['target'], df['mu'], df['alpha'])
    u = np.random.default_rng(0).uniform(size=len(df))
    return df['lo'].to_numpy() + u * (df['hi'].to_numpy() - df['lo'].to_numpy())


def quantile_ranks(df: pd.DataFrame, quantiles: list[float]) -> pd.Series:
    """
    Fraction of predicted quantiles strictly below the target, per row. A
    discrete PIT: histogram it; a U shape means intervals too narrow, a hump
    too wide, a slope a biased median.
    """
    Q = _quantile_matrix(df, quantiles)
    y = df['target'].to_numpy(dtype=float)[:, None]
    return pd.Series((Q < y).mean(axis=1), index=df.index, name='quantile_rank')


def summarize_intervals(df: pd.DataFrame,
                        quantiles: list[float],
                        group_cols: Iterable[str] | None = None) -> pd.DataFrame:
    """
    WIS (with decomposition), coverage and width, pooled or per group.

    Returns one row per (group, interval) with columns: group columns,
    nominal, coverage, below, above, width, wis, dispersion, underprediction,
    overprediction, n.
    """
    group_cols = list(group_cols or [])
    scores = wis(df, quantiles, decompose=True)
    data   = pd.concat([df, scores], axis=1)

    def _one(g: pd.DataFrame) -> pd.DataFrame:
        cw = coverage_and_width(g, quantiles)
        for c in ['wis', 'dispersion', 'underprediction', 'overprediction']:
            cw[c] = float(g[c].mean())
        cw['n'] = len(g)
        return cw

    if not group_cols:
        return _one(data)

    parts = []
    for key, g in data.groupby(group_cols, sort=True):
        out = _one(g)
        key = key if isinstance(key, tuple) else (key,)
        for c, v in zip(group_cols, key):
            out.insert(0, c, v)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# seasons
# --------------------------------------------------------------------------- #
# Default season windows (ISO weeks, inclusive, may wrap around new year).
DEFAULT_SEASONS: dict[str, tuple[int, int]] = {
    'influenza':     (40, 15),
    'norovirus':     (40, 15),
    'chickenpox':    (1, 26),
    'campylobacter': (22, 40),
}


def season_weeks_for(disease: str | None) -> tuple[int, int]:
    """Season window for a disease; falls back to winter (weeks 40-15)."""
    return DEFAULT_SEASONS.get(str(disease).lower(), (40, 15))


def season_mask(timestamps: pd.Series, start_week: int, end_week: int) -> pd.Series:
    """True for ISO weeks in [start_week, end_week], wrapping around new year."""
    week = pd.to_datetime(timestamps).dt.isocalendar().week.astype(int)
    if start_week <= end_week:
        return (week >= start_week) & (week <= end_week)
    return (week >= start_week) | (week <= end_week)


def model_prediction_frame(model,
                           dataset: str = 'test',
                           horizon: int = 0,
                           is_original: bool = True,
                           season: str | None = None,
                           season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    A model's stored predictions for one horizon, optionally filtered by season.

    season : None (all weeks), 'in' or 'off'. The window comes from
        ``season_weeks`` or, if None, from the model's disease
        (``DEFAULT_SEASONS``). Weeks refer to the target time.
    """
    df = model.predictions.get_preds(dataset).get(horizon, is_original, False)
    if season is None:
        return df
    if season not in ('in', 'off'):
        raise ValueError("season must be None, 'in' or 'off'")
    weeks = season_weeks or season_weeks_for(getattr(model.epiconfig, 'disease', None))
    mask  = season_mask(df[model.epiconfig.temporal_column], *weeks)
    return df[mask if season == 'in' else ~mask].reset_index(drop=True)


def evaluate_model_intervals(model,
                             dataset: str = 'test',
                             is_original: bool = True,
                             group_cols: Iterable[str] | None = None,
                             season: str | None = None,
                             season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    ``summarize_intervals`` for every horizon of a forecasted model, on the
    original scale by default. Adds a ``horizon`` column.

    ``group_cols`` may include the node column (``EpiConfig.id_column``) for
    per-node coverage, or a column you merged in yourself (e.g. a size class).

    ``season='in'`` scores only the epidemic season (see ``model_prediction_frame``).
    Off-season weeks with zero counts are covered by almost any interval and
    otherwise dominate the pooled coverage.
    """
    quantiles = model.epiconfig.quantiles
    if quantiles is None:
        raise ValueError('model was not run in interval mode (EpiConfig.quantiles is None)')

    coll  = model.predictions.get_preds(dataset)
    parts = []
    for hh in coll.horizons:
        df  = model_prediction_frame(model, dataset, hh, is_original, season, season_weeks)
        out = summarize_intervals(df, quantiles, group_cols)
        out.insert(0, 'horizon', hh)
        parts.append(out)
    res = pd.concat(parts, ignore_index=True)
    res.insert(0, 'model', model.name)
    if season is not None:
        res.insert(1, 'season', season)
    return res


def compare_models(models,
                   dataset: str = 'test',
                   season: str | None = 'in',
                   reference: str | None = None,
                   season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    One row per model and horizon: WIS (and parts), 50/80/95% coverage and width,
    and for models with an NB predictive distribution the count-aware coverage
    from the randomised PIT (``pitcov50`` ...). For count forecasts read pitcov,
    not cov: whole-number quantiles make plain interval coverage too high.

    ``models`` is a list or a dict {label: model}. With ``reference`` (a model
    name or label), adds ``rel_wis`` = WIS / WIS of the reference (< 1 is better).
    """
    items = models.items() if isinstance(models, dict) else [(m.name, m) for m in models]
    rows  = []
    for label, m in items:
        res = evaluate_model_intervals(m, dataset, season=season, season_weeks=season_weeks)
        for hh, g in res.groupby('horizon'):
            row = {'model': label, 'horizon': hh}
            for c in ['wis', 'dispersion', 'underprediction', 'overprediction', 'n']:
                row[c] = g[c].iloc[0]
            for _, r in g.iterrows():
                lvl = int(round(r['nominal'] * 100))
                row[f'cov{lvl}']   = r['coverage']
                row[f'width{lvl}'] = r['width']
            pit = model_pit(m, dataset, season, season_weeks) if hh == 0 else None
            if pit is not None and len(pit):
                for _, r in pit_coverage(pit, m.epiconfig.quantiles).iterrows():
                    row[f'pitcov{int(round(r["nominal"] * 100))}'] = r['coverage']
            rows.append(row)
    out = pd.DataFrame(rows)
    if reference is not None:
        ref = out[out['model'] == reference].set_index('horizon')['wis']
        if ref.empty:
            raise ValueError(f'reference {reference!r} not among {out["model"].unique().tolist()}')
        out['rel_wis'] = out['wis'] / out['horizon'].map(ref)
    return out
