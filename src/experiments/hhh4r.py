"""
The hhh4 reference model, fitted in R (package ``surveillance``) and used from
Python like any other model.

Python writes the counts, the graph, population sizes and the specification to
a folder, runs ``r/hhh4_fit.R`` with ``Rscript``, and reads the results back
into an :class:`HHH4RModel`. That object has the same interface the comparison
and diagnostics code uses (``name``, ``epiconfig``, ``predictions``), so it
appears in ``compare_models``, ``sanity_report`` and the calibration plots next
to the GNN and the baselines.

Requirements (once): R, and in R ``install.packages("surveillance")``. If
``Rscript`` is not on the PATH, set ``hhh4_r.rscript`` in the config to its full
path (on Windows e.g. ``C:/Program Files/R/R-4.4.1/bin/Rscript.exe``).

Forecasts: hhh4 is fitted on one-week-ahead data; lead-L forecasts come from
simulating the fitted model forward L weeks from each forecast origin, which is
how hhh4 forecasts are normally made. Components (endemic / epidemic /
neighbourhood) are the ONE-STEP-AHEAD means at each target week given the
observed past, which is what hhh4 defines; compare them with the GNN at lead 1.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).resolve().parent / 'r' / 'hhh4_fit.R'

DEFAULT_SPEC = dict(nsim=500, seed=1, harmonics=1, max_lag=5, random_effects=True,
                    power_law=True, family='NegBin1')


class RNotAvailable(RuntimeError):
    pass


def find_rscript(rscript: str | None = None) -> str:
    exe = rscript or 'Rscript'
    path = shutil.which(exe) or (exe if Path(exe).exists() else None)
    if path is None:
        raise RNotAvailable(
            f'Rscript not found ({exe!r}). Install R and, in R, install.packages("surveillance"); '
            'or set hhh4_r.rscript in the config to the full path of Rscript.')
    return path


# --------------------------------------------------------------------------- #
# export -> R -> import
# --------------------------------------------------------------------------- #
def run_hhh4_r(counts: pd.DataFrame,
               adjacency: np.ndarray,
               population: np.ndarray,
               fit_end: pd.Timestamp,
               t0_dates,
               lead: int,
               quantiles: list[float],
               folder: str | Path,
               spec: dict | None = None,
               rscript: str | None = None,
               timeout: int = 7200) -> dict[str, pd.DataFrame]:
    """
    Fit hhh4 in R and return its outputs as DataFrames.

    Parameters
    ----------
    counts : DataFrame, index = weekly dates (sorted), columns = nodes 0..N-1
    adjacency : [N, N] array; > 0 means neighbours
    population : [N] population sizes (endemic offset)
    fit_end : last date used for fitting (the model never sees later counts)
    t0_dates : forecast origins (dates in ``counts.index``)
    lead : weeks ahead
    """
    exe = find_rscript(rscript)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec = {**DEFAULT_SPEC, **(spec or {})}

    counts = counts.sort_index()
    dates = pd.to_datetime(counts.index)
    pos = {d: i + 1 for i, d in enumerate(dates)}                 # 1-based rows in R
    t0_rows = [pos[pd.Timestamp(d)] for d in t0_dates if pd.Timestamp(d) in pos]
    if not t0_rows:
        raise ValueError('none of the forecast origins is in the count table')
    fit_end_row = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(fit_end)), side='right'))
    if fit_end_row < 10:
        raise ValueError('too few weeks before fit_end to fit hhh4')

    out = counts.copy()
    out.index = dates.strftime('%Y-%m-%d')
    out.columns = [f'node_{i}' for i in range(out.shape[1])]
    out.round().astype(int).to_csv(folder / 'counts.csv', index_label='date')
    pd.DataFrame((np.asarray(adjacency) > 0).astype(int)).to_csv(folder / 'adjacency.csv', header=False, index=False)
    pd.DataFrame({'node': range(len(population)), 'population': np.asarray(population, float)}) \
        .to_csv(folder / 'population.csv', index=False)

    kv = {**spec, 'fit_end_row': fit_end_row, 'lead': int(lead),
          't0_rows': ' '.join(map(str, t0_rows)), 'quantiles': ' '.join(map(str, quantiles))}
    pd.DataFrame({'key': list(kv), 'value': [str(v) for v in kv.values()]}).to_csv(folder / 'spec.csv', index=False)

    proc = subprocess.run([exe, str(R_SCRIPT), str(folder)], capture_output=True, text=True, timeout=timeout)
    (folder / 'r_log.txt').write_text(proc.stdout + '\n--- stderr ---\n' + proc.stderr)
    if proc.returncode != 0:
        tail = '\n'.join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f'hhh4 in R failed (see {folder / "r_log.txt"}):\n{tail}')

    res = {name: pd.read_csv(folder / f'{name}.csv')
           for name in ('predictions', 'components', 'coefficients', 'unit_effects', 'fit_info')}
    res['dates'] = pd.Series(dates)
    return res


# --------------------------------------------------------------------------- #
# model wrapper
# --------------------------------------------------------------------------- #
class _OneCollection:
    """Minimal stand-in for PredictionCollection (test split, horizon 0, original scale)."""
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.horizons = [0]

    def get(self, horizon: int, is_original: bool, spatially_aggregated: bool) -> pd.DataFrame:
        if horizon != 0 or not is_original or spatially_aggregated:
            raise KeyError('the hhh4 reference model stores horizon 0, original scale, per node')
        return self._df.copy()


class _Predictions:
    def __init__(self, df: pd.DataFrame):
        self._test = _OneCollection(df)

    def get_preds(self, dataset: str):
        if dataset != 'test':
            raise KeyError('the hhh4 reference model only forecasts the test split')
        return self._test


class HHH4RModel:
    """
    hhh4 fitted in R, presented like the Python models.

    Attributes
    ----------
    name, epiconfig, predictions : as for the other models
    components : one-step components per target week and node
    coefficients, unit_effects, fit_info : parameter tables from R
    """
    def __init__(self, res: dict, epiconfig, name: str = 'hhh4_R'):
        self.name = name
        self.epiconfig = epiconfig
        dates = res['dates']
        lead = int(epiconfig.horizon_leadtime)
        t_col, id_col = epiconfig.temporal_column, epiconfig.id_column
        q = list(epiconfig.quantiles)

        p = res['predictions']
        target_time = dates.iloc[(p['t0_row'] - 1 + lead).to_numpy()].to_numpy()
        frame = pd.DataFrame({t_col: target_time, id_col: p['node'].astype(int), 'target': p['target'].astype(float)})
        for i in range(len(q)):
            frame[f'pred_q{i + 1}'] = p[f'q_{i + 1}'].astype(float)
        self._frame = frame
        self._pit_bounds = pd.DataFrame({'target_time': target_time, 'lo': p['cdf_lo'], 'hi': p['cdf_hi']})
        self.predictions = _Predictions(frame)

        c = res['components'].copy()
        c[t_col] = dates.iloc[(c['row'] - 1).to_numpy()].to_numpy()
        c = c.rename(columns={'node': id_col, 'mean': 'mu'}).drop(columns='row')
        for k in ('endemic', 'epidemic', 'neighbourhood'):
            c[f'share_{k}'] = c[k] / c['mu'].where(c['mu'] > 0)
        self.components = c
        self.coefficients = res['coefficients']
        self.unit_effects = res['unit_effects']
        self.fit_info = dict(zip(res['fit_info']['key'], res['fit_info']['value']))

    # randomised PIT from the simulated predictive distribution
    def pit_bounds(self, dataset: str = 'test') -> pd.DataFrame:
        return self._pit_bounds.copy()

    def component_table(self, season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
        """One-step component shares (mu-weighted), all / in-season / off-season."""
        from ..models.utils.intervalmetrics import season_mask, season_weeks_for
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
                         **{f'share_{k}': g[k].sum() / max(tot, 1e-12)
                            for k in ('endemic', 'epidemic', 'neighbourhood')}})
        return pd.DataFrame(rows)


def epiconfig_view(epiconfig) -> SimpleNamespace:
    """The few EpiConfig fields the comparison code reads."""
    return SimpleNamespace(**{k: getattr(epiconfig, k) for k in
                              ('quantiles', 'horizon_leadtime', 'horizon_size', 'temporal_column',
                               'id_column', 'disease', 'level', 'pred_column', 'target_column')
                              if hasattr(epiconfig, k)})


# --------------------------------------------------------------------------- #
# glue for the experiment runner (real data pipeline)
# --------------------------------------------------------------------------- #
def pipeline_counts(edo) -> tuple[pd.DataFrame, np.ndarray]:
    """Weekly case counts [dates x nodes] and mean population per node from the pipeline."""
    cfg = edo.config
    t_col, id_col = cfg.temporal_column, cfg.id_column
    if cfg.target_column != 'cases':
        raise ValueError("hhh4 needs target_column='cases'")
    n = edo.data_context.num_nodes
    epi = edo.data_processed.epidata
    counts = epi.pivot_table(index=t_col, columns=id_col, values='cases', aggfunc='sum').sort_index()
    counts = counts.reindex(columns=range(n)).fillna(0)
    pop = edo.data_context.population_size
    pop = pop.groupby(id_col)['population_size'].mean().reindex(range(n))
    return counts, pop.fillna(pop.mean()).to_numpy()


def hhh4_r_from_pipeline(edo, graph, reference_model, folder, spec=None, rscript=None) -> HHH4RModel:
    """
    Fit hhh4 on the same data, graph and forecast rows as ``reference_model``
    (a forecasted HHH4Model): counts from the processed data, fit on everything
    before the test split, forecast origins = the reference model's test rows.
    """
    cfg = edo.config
    t_col = cfg.temporal_column
    counts, pop = pipeline_counts(edo)

    lead = int(cfg.horizon_leadtime)
    ref = reference_model.predictions.get_preds('test').get(0, True, False)
    t0_dates = sorted(pd.to_datetime(ref[t_col]).unique() - pd.Timedelta(weeks=lead))

    # fit on all weeks before the first test target; the model then only uses
    # observed counts up to each forecast origin when forecasting
    fit_end = pd.Timestamp(edo.data_context.temporal_summary.split_valtest) - pd.Timedelta(days=1)

    res = run_hhh4_r(counts, graph.adjacency_matrix.cpu().numpy(), pop, fit_end, t0_dates, lead,
                     list(cfg.quantiles), folder, spec=spec, rscript=rscript)
    return HHH4RModel(res, epiconfig_view(cfg))
