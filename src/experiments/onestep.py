"""
One-step models with simulated multi-week forecasts, in the real-data pipeline.

hhh4 is fitted one week ahead and forecasts L weeks ahead by simulating the
fitted process forward; its components are one-step transmission routes. This
module does the same for

- ``hhh4_py``: hhh4 in Python (:class:`HHH4Py`, no R needed), and
- the neural model trained at lead 1 (``neural_hhh4_sim``): each simulated week
  is fed back into the input window of the next step,

and wraps both as :class:`SampleForecastModel`, which ``compare_models``, the
randomised PIT and the calibration plots treat like the other models.

Why: a model trained directly L weeks ahead learns *prediction weights* (how
much of the count in L weeks is best predicted from the own region, the
neighbours or the background); on simulated data with a known split those
weights are a worse estimate of the routes of transmission than a one-step
model's (``src/experiments/attribution.py``).
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

COMPS = ('endemic', 'epidemic', 'neighbourhood')


# --------------------------------------------------------------------------- #
# generic wrapper for forecasts given as predictive draws
# --------------------------------------------------------------------------- #
class _Collection:
    def __init__(self, df):
        self._df = df
        self.horizons = [0]

    def get(self, horizon, is_original, spatially_aggregated):
        if horizon != 0 or not is_original or spatially_aggregated:
            raise KeyError('simulation-based models store horizon 0, original scale, per node')
        return self._df.copy()


class _Preds:
    def __init__(self, df):
        self._c = _Collection(df)

    def get_preds(self, dataset):
        if dataset != 'test':
            raise KeyError('simulation-based models only forecast the test split')
        return self._c


class SampleForecastModel:
    """
    A model whose test forecasts are predictive draws, presented like the other
    models (``name``, ``epiconfig``, ``predictions``, ``pit_bounds``).

    Parameters
    ----------
    draws : [O, S, N] draws at the target weeks
    y_true : [O, N] observed counts at the target weeks
    target_dates : [O] target weeks
    components : optional one-step components (columns: target time, node,
        endemic, epidemic, neighbourhood, mu, target) for ``component_table``
    """
    def __init__(self, name, epiconfig, draws, y_true, target_dates, components=None, info=None):
        self.name = name
        self.epiconfig = epiconfig
        t_col, id_col = epiconfig.temporal_column, epiconfig.id_column
        q = list(epiconfig.quantiles)
        O, S, N = draws.shape
        qv = np.quantile(draws, q, axis=1, method='inverted_cdf')                  # [Q, O, N]
        tt = np.repeat(pd.to_datetime(np.asarray(target_dates)), N)
        frame = pd.DataFrame({t_col: tt, id_col: np.tile(np.arange(N), O), 'target': y_true.ravel().astype(float)})
        for i in range(len(q)):
            frame[f'pred_q{i + 1}'] = qv[i].ravel()
        self._frame = frame
        yt = y_true[:, None, :]
        self._pit = pd.DataFrame({'target_time': tt, 'lo': (draws <= yt - 1).mean(1).ravel(),
                                  'hi': (draws <= yt).mean(1).ravel()})
        self.mean_forecast = draws.mean(1)
        self.predictions = _Preds(frame)
        self.components = components
        self.info = info or {}

    def pit_bounds(self, dataset: str = 'test') -> pd.DataFrame:
        return self._pit.copy()

    def component_table(self, season_weeks=None) -> pd.DataFrame:
        """One-step component shares (mu-weighted): all / in-season / off-season."""
        from ..models.utils.intervalmetrics import season_mask, season_weeks_for
        if self.components is None:
            raise ValueError(f'{self.name} has no components')
        c = self.components
        weeks = season_weeks or season_weeks_for(getattr(self.epiconfig, 'disease', None))
        ins = season_mask(c[self.epiconfig.temporal_column], *weeks).to_numpy()
        rows = []
        for period, g in [('all', c), ('in-season', c[ins]), ('off-season', c[~ins])]:
            if len(g) == 0:
                continue
            tot = g['mu'].sum()
            rows.append({'model': self.name, 'period': period, 'n_rows': len(g),
                         'bias_ratio_one_step': tot / max(g['target'].sum(), 1e-12),
                         **{f'share_{k}': g[k].sum() / max(tot, 1e-12) for k in COMPS}})
        return pd.DataFrame(rows)


def _view(epiconfig, lead: int | None = None) -> SimpleNamespace:
    keys = ('quantiles', 'horizon_leadtime', 'horizon_size', 'temporal_column', 'id_column', 'disease',
            'level', 'pred_column', 'target_column')
    v = SimpleNamespace(**{k: getattr(epiconfig, k) for k in keys if hasattr(epiconfig, k)})
    if lead is not None:
        v.horizon_leadtime = int(lead)
    return v


def reference_origins(reference, lead: int) -> list[pd.Timestamp]:
    """Forecast origins (target week - lead) of a forecasted reference model's test rows."""
    t_col = reference.epiconfig.temporal_column
    ref = reference.predictions.get_preds('test').get(0, True, False)
    return sorted(pd.to_datetime(ref[t_col]).unique() - pd.Timedelta(weeks=lead))


# --------------------------------------------------------------------------- #
# hhh4 in Python on the pipeline data
# --------------------------------------------------------------------------- #
def fit_hhh4_py(counts: pd.DataFrame, adjacency, population, fit_end, spec: dict | None = None):
    """Fit :class:`HHH4Py` on all target weeks up to ``fit_end``."""
    from ..models.statistical import HHH4Py
    spec = spec or {}
    dates = pd.to_datetime(counts.index)
    n_fit = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(fit_end)), side='right'))
    if n_fit < 20:
        raise ValueError('too few weeks before fit_end to fit hhh4')
    h = HHH4Py(harmonics=int(spec.get('harmonics', 1)), random_effects=bool(spec.get('random_effects', True)),
               power_law=bool(spec.get('power_law', True)), max_lag=int(spec.get('max_lag', 5)))
    return h.fit(counts.to_numpy(dtype=float), np.asarray(adjacency), np.asarray(population, float),
                 fit_rows=np.arange(1, n_fit))


def hhh4_py_model(counts: pd.DataFrame, adjacency, population, fit_end, origins, lead: int, epiconfig,
                  spec: dict | None = None, name: str = 'hhh4_py') -> SampleForecastModel:
    """hhh4 (Python) fitted up to ``fit_end``, forecasting ``lead`` weeks ahead by simulation."""
    spec = spec or {}
    counts = counts.sort_index()
    dates = pd.to_datetime(counts.index)
    pos = {d: i for i, d in enumerate(dates)}
    h = fit_hhh4_py(counts, adjacency, population, fit_end, spec)
    o_rows = np.array([pos[pd.Timestamp(o)] for o in origins if pd.Timestamp(o) in pos
                       and pos[pd.Timestamp(o)] + lead < len(dates)])
    if len(o_rows) == 0:
        raise ValueError('none of the forecast origins is in the count table')
    draws = h.simulate(o_rows, lead, nsim=int(spec.get('nsim', 500)), seed=int(spec.get('seed', 1)))
    y = counts.to_numpy(dtype=float)
    c = h.fitted_components(o_rows + lead)
    t_col, id_col = epiconfig.temporal_column, epiconfig.id_column
    comp = pd.DataFrame({t_col: dates[c['row'].to_numpy()], id_col: c['node'].to_numpy(),
                         **{k: c[k].to_numpy() for k in COMPS}, 'mu': c['mean'].to_numpy(),
                         'target': c['target'].to_numpy()})
    m = SampleForecastModel(name, _view(epiconfig, lead), draws, y[o_rows + lead], dates[o_rows + lead],
                            components=comp, info=h.summary())
    m.hhh4 = h
    m.coefficients = pd.DataFrame({'name': list(m.info), 'estimate': list(m.info.values())})
    m.unit_effects = h.node_parameters()
    return m


# --------------------------------------------------------------------------- #
# neural model trained one week ahead, forecasting by simulation
# --------------------------------------------------------------------------- #
def _snapshots_by_t0(model) -> dict:
    out = {}
    for ds in ('train', 'val', 'test'):
        snaps = list(model._get_dataloader(ds))
        if not snaps:
            continue
        for t0, s in zip(pd.to_datetime(model._t0_timestamps(ds, len(snaps))), snaps):
            out[pd.Timestamp(t0)] = s
    return out


def _lag_of(feature: str) -> int:
    try:
        return int(feature.rsplit('_lag', 1)[1])
    except (IndexError, ValueError):
        return 0


def simulate_neural_pipeline(model, origins, lead: int, nsim: int = 200, seed: int = 0):
    """
    Forward simulation with a lead-1 HHH4Model of the pipeline. For each origin
    the input window of step k is the observed snapshot at origin + k - 1 with the
    case features of weeks after the origin replaced by the simulated counts
    (other features, e.g. the season, are known in advance and kept).

    Returns (draws [O, nsim, N], used origins).
    """
    if int(model.epiconfig.horizon_leadtime) != 1:
        raise ValueError('simulate_neural_pipeline needs a model trained at lead 1')
    idx = [int(i) for i in model.model.incidence_idx.cpu().tolist()]     # same order as incidence_features
    inc = [(f, _lag_of(name)) for f, name in zip(idx, model.incidence_features)]
    S = int(model.epiconfig.sequence_length)
    snaps = _snapshots_by_t0(model)
    scale = float(getattr(model, 'alpha_scale', 1.0))
    module = model.model
    module.eval()
    rng = np.random.default_rng(seed)
    week = pd.Timedelta(weeks=1)

    draws, used = [], []
    with torch.no_grad():
        for o in map(pd.Timestamp, origins):
            if any(o + k * week not in snaps for k in range(lead)):
                continue
            sim: dict[pd.Timestamp, np.ndarray] = {}
            for k in range(lead):
                t_last = o + k * week
                snap = snaps[t_last].to(model.device)
                x0 = snap.x
                ei, ew = snap.graph.edge_index, snap.graph.edge_weight
                out = np.empty((nsim, x0.shape[0]))
                for s in range(nsim):
                    x = x0.clone()
                    for f, lag in inc:
                        for pos in range(S):
                            w = t_last - (S - 1 - pos) * week - lag * week
                            if w > o:
                                x[:, f, pos] = torch.as_tensor(sim[w][s], dtype=x.dtype, device=x.device)
                    mu, alpha = module(x, ei, ew)
                    mu = mu[:, 0].cpu().numpy()
                    a = alpha[:, 0].cpu().numpy() * scale
                    out[s] = rng.poisson(rng.gamma(1.0 / a, mu * a))
                sim[t_last + week] = out
            draws.append(sim[o + lead * week])
            used.append(o)
    if not used:
        raise ValueError('no origin has the snapshots needed for simulation')
    if len(used) < len(origins):
        warnings.warn(f'{len(origins) - len(used)} origins skipped (snapshots missing)')
    return np.stack(draws), used


def neural_sim_model(model1, counts: pd.DataFrame, origins, lead: int, epiconfig, nsim: int = 200,
                     seed: int = 0, name: str = 'neural_hhh4_sim') -> SampleForecastModel:
    """Lead-L forecasts of a lead-1 neural model by simulation, plus its one-step components."""
    draws, used = simulate_neural_pipeline(model1, origins, lead, nsim=nsim, seed=seed)
    counts = counts.sort_index()
    idx = pd.to_datetime(counts.index)
    targets = pd.DatetimeIndex(used) + pd.Timedelta(weeks=lead)
    y = counts.reindex(targets).to_numpy(dtype=float)
    keep = ~np.isnan(y).any(1)
    t_col, id_col = epiconfig.temporal_column, epiconfig.id_column
    fc = model1.forecast_components('test')
    fc = fc[fc['horizon'] == 0]
    comp = pd.DataFrame({t_col: pd.to_datetime(fc[t_col]) + pd.Timedelta(weeks=1), id_col: fc[id_col].to_numpy(),
                         **{k: fc[k].to_numpy() for k in COMPS}, 'mu': fc['mu'].to_numpy(),
                         'target': fc['target'].to_numpy()})
    m = SampleForecastModel(name, _view(epiconfig, lead), draws[keep], y[keep], targets[keep], components=comp)
    m.one_step_model = model1
    return m
