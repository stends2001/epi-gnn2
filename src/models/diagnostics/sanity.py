"""
Automated sanity checks for probabilistic forecasts.

``sanity_report(model, baselines)`` runs a list of checks and marks each PASS,
WARN or FAIL, with a short explanation. The aim is to catch forecasts that look
reasonable in a plot but are not useful, for example:

- a median that copies the last observation (lag check),
- systematic under- or over-prediction (bias ratio),
- intervals that only look calibrated because of off-season zeros
  (in-season coverage),
- a model that does not beat simple baselines (relative WIS),
- for ``HHH4Model``: components that do not add up, or a branch that is never used.

All checks use horizon 0, the original scale, and by default the epidemic
season of the model's disease.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.intervalmetrics import (
    model_prediction_frame, summarize_intervals, quantile_ranks, season_weeks_for,
)

PASS, WARN, FAIL, INFO = 'PASS', 'WARN', 'FAIL', 'INFO'


def _median_col(model) -> str:
    q = model.epiconfig.quantiles
    return f'pred_q{len(q) // 2 + 1}' if q else model.epiconfig.pred_column


def lag_correlation(model,
                    dataset: str = 'test',
                    max_lag: int | None = None,
                    season: str | None = None) -> pd.DataFrame:
    """
    Correlation between the median forecast at target week t and the truth at
    week t - k, for k = 0 .. max_lag (log1p scale, pooled over nodes).

    A good forecast correlates best at k = 0. A forecast that copies the last
    observation correlates best at k = lead time: it is late by exactly the lead.
    """
    df = model_prediction_frame(model, dataset, 0, True, None)
    t_col, id_col = model.epiconfig.temporal_column, model.epiconfig.id_column
    med = _median_col(model)
    lead = model.epiconfig.horizon_leadtime
    max_lag = max_lag if max_lag is not None else lead + 3

    df = df.sort_values([id_col, t_col]).copy()
    if season is not None:
        from ..utils.intervalmetrics import season_mask
        weeks = season_weeks_for(model.epiconfig.disease)
        m = season_mask(df[t_col], *weeks)
        keep = m if season == 'in' else ~m
    else:
        keep = pd.Series(True, index=df.index)

    rows = []
    for k in range(0, max_lag + 1):
        past = df.groupby(id_col)['target'].shift(k)
        ok = keep & past.notna() & df[med].notna()
        if ok.sum() < 10:
            continue
        r = np.corrcoef(np.log1p(df.loc[ok, med].clip(lower=0)),
                        np.log1p(past[ok].clip(lower=0)))[0, 1]
        rows.append({'lag_weeks': k, 'corr': r, 'n': int(ok.sum())})
    return pd.DataFrame(rows)


def _status_range(value: float, good: tuple[float, float], ok: tuple[float, float]) -> str:
    if good[0] <= value <= good[1]:
        return PASS
    if ok[0] <= value <= ok[1]:
        return WARN
    return FAIL


def sanity_report(model,
                  baselines=None,
                  dataset: str = 'test',
                  season: str | None = 'in',
                  season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    Run all checks and return a table: check, value, expected, status, note.

    Parameters
    ----------
    model : forecasted model in interval mode
    baselines : list or dict {label: model} of forecasted baselines on the SAME
        config (e.g. Persistence and SeasonalAverage), for relative WIS.
    season : 'in' (default), 'off' or None (all weeks) for the scoring checks.
    """
    rows: list[dict] = []

    def add(check, value, expected, status, note=''):
        rows.append({'check': check, 'value': value, 'expected': expected,
                     'status': status, 'note': note})

    q = model.epiconfig.quantiles
    if q is None:
        raise ValueError('sanity_report expects interval forecasts (EpiConfig.quantiles set).')
    med = _median_col(model)

    full = model_prediction_frame(model, dataset, 0, True, None)
    df   = model_prediction_frame(model, dataset, 0, True, season, season_weeks)
    pred_cols = [f'pred_q{i+1}' for i in range(len(q))]

    # ---- 1. basic integrity ----
    n_nan = int(full[pred_cols + ['target']].isna().sum().sum())
    add('missing values in predictions', n_nan, '0', PASS if n_nan == 0 else FAIL)

    crossing = float((np.diff(full[pred_cols].to_numpy(), axis=1) < -1e-9).any(axis=1).mean())
    add('rows with crossing quantiles', round(crossing, 4), '0', PASS if crossing == 0 else FAIL)

    neg = float((full[pred_cols] < 0).any(axis=1).mean())
    add('rows with negative quantiles', round(neg, 4), '0 for counts / rates',
        PASS if neg == 0 else WARN)

    # ---- 2. level and shape of the point forecast ----
    ratio = df[med].sum() / max(df['target'].sum(), 1e-12)
    add('bias ratio (sum median / sum truth)', round(ratio, 3), '0.8 - 1.25',
        _status_range(ratio, (0.8, 1.25), (0.6, 1.6)),
        'below 1: under-predicts. For skewed NB forecasts the median sits below '
        'the mean, so a ratio somewhat below 1 is expected; see the mu check.')

    if len(df) > 2:
        r = np.corrcoef(np.log1p(df[med].clip(lower=0)), np.log1p(df['target'].clip(lower=0)))[0, 1]
        add('correlation median vs truth (log1p)', round(r, 3), '>= 0.7',
            PASS if r >= 0.7 else WARN if r >= 0.4 else FAIL)

    lags = lag_correlation(model, dataset)
    if not lags.empty:
        best = int(lags.loc[lags['corr'].idxmax(), 'lag_weeks'])
        lead = model.epiconfig.horizon_leadtime
        c0   = float(lags.loc[lags['lag_weeks'] == 0, 'corr'].iloc[0]) if (lags['lag_weeks'] == 0).any() else np.nan
        cl   = float(lags.loc[lags['lag_weeks'] == lead, 'corr'].iloc[0]) if (lags['lag_weeks'] == lead).any() else np.nan
        status = PASS if best == 0 else WARN
        add('lag check: truth lag that best matches the median', best,
            '0 (forecast is on time)', status,
            f'corr at lag 0 = {c0:.2f}, at lag {lead} (= lead) = {cl:.2f}. Best at or near '
            'the lead means the median tracks the last observation. For strongly '
            'autoregressive series even a good forecast does this; it is a problem only '
            'when the model also fails to beat Persistence (see relative WIS).')

    # ---- 3. calibration ----
    if len(df) > 0:
        from ..utils.intervalmetrics import model_pit, pit_coverage
        summ = summarize_intervals(df, q)
        pit  = model_pit(model, dataset, season, season_weeks)
        if pit is not None and len(pit):
            # count forecasts: judge calibration on the randomised PIT; plain interval
            # coverage of whole-number quantiles is biased upwards and only reported
            for _, r in pit_coverage(pit, q).iterrows():
                gap = r['coverage'] - r['nominal']
                add(f"coverage {int(round(r['nominal']*100))}% (randomised PIT)", round(r['coverage'], 3),
                    f"{r['nominal']:.2f} ± 0.05",
                    PASS if abs(gap) <= 0.05 else WARN if abs(gap) <= 0.15 else FAIL,
                    f"below {r['below']:.3f}, above {r['above']:.3f} (each about (1 - nominal) / 2)")
            for _, r in summ.iterrows():
                add(f"coverage {int(round(r['nominal']*100))}% interval (whole-number quantiles)",
                    round(r['coverage'], 3), 'above nominal for small counts', INFO,
                    'count quantiles are whole numbers, so intervals hold more than nominal')
        else:
            for _, r in summ.iterrows():
                gap = r['coverage'] - r['nominal']
                add(f"coverage {int(round(r['nominal']*100))}% interval", round(r['coverage'], 3),
                    f"{r['nominal']:.2f} ± 0.05",
                    PASS if abs(gap) <= 0.05 else WARN if abs(gap) <= 0.15 else FAIL,
                    f"below {r['below']:.3f}, above {r['above']:.3f} "
                    "(should each be about (1 - nominal) / 2)")

        widths = df[pred_cols[-1]] - df[pred_cols[0]]
        zero_w = float((widths <= 1e-9).mean())
        add('zero-width outer intervals', round(zero_w, 3), '< 0.2',
            PASS if zero_w < 0.2 else WARN,
            'many zero-width intervals mean coverage is decided by exact hits on 0')

        ranks = quantile_ranks(df, q)
        extreme = float(((ranks == 0) | (ranks == 1)).mean())
        if pit is not None and len(pit):
            extreme = float(((pit < q[0]) | (pit > 1 - q[0])).mean())
        expected = 2 * q[0]
        add('truth outside all quantiles (PIT tails)', round(extreme, 3), f'about {expected:.3f}',
            PASS if extreme <= 2 * expected + 0.02 else WARN,
            'much larger: intervals too narrow; much smaller: too wide')

    # ---- 4. skill against baselines ----
    if baselines:
        items = baselines.items() if isinstance(baselines, dict) else [(b.name, b) for b in baselines]
        from ..utils.intervalmetrics import wis as _wis
        model_wis = float(_wis(df, q).mean()) if len(df) else np.nan
        add('WIS (this model)', round(model_wis, 4), 'lower is better', INFO)
        for label, b in items:
            bdf = model_prediction_frame(b, dataset, 0, True, season, season_weeks)
            bw  = float(_wis(bdf, q).mean())
            rel = model_wis / bw if bw > 0 else np.nan
            add(f'relative WIS vs {label}', round(rel, 3), '< 1',
                PASS if rel < 0.98 else WARN if rel < 1.05 else FAIL,
                f'{label} WIS = {bw:.4f}. Same config is required for a fair comparison.')

    # ---- 5. HHH4-specific ----
    if hasattr(model, 'forecast_components'):
        from .components import component_table
        ct = component_table(model, dataset, season_weeks)
        comp = model.forecast_components(dataset)
        total = comp[['endemic', 'epidemic', 'neighbourhood']].sum(axis=1)
        err = float((total - comp['mu']).abs().max())
        add('components add up to mu', f'{err:.2e}', '< 1e-3', PASS if err < 1e-3 else FAIL)

        period = 'in-season' if season == 'in' else 'all'
        row = ct[(ct['period'] == period) & (ct['horizon'] == 0)]
        if not row.empty:
            row = row.iloc[0]
            add('bias ratio of mu (sum mu / sum truth)', round(float(row['bias_ratio']), 3), '0.8 - 1.25',
                _status_range(row['bias_ratio'], (0.8, 1.25), (0.6, 1.6)),
                'the NB mean; should be closer to 1 than the median ratio')
            for c in ('endemic', 'epidemic', 'neighbourhood'):
                v = row[f'share_{c}']
                add(f'{c} share of mu ({period})', round(float(v), 3), '> 0.02 (branch used)',
                    PASS if v > 0.02 else WARN,
                    'a branch near 0 contributes nothing; check its inputs' if v <= 0.02 else '')

        alpha = float(comp['alpha'].mean())
        add('NB dispersion alpha (mean)', round(alpha, 4), '0.01 - 2',
            PASS if 0.01 <= alpha <= 2 else WARN,
            'very large: the mean explains little; very small: close to Poisson, '
            'check that coverage still holds')

    out = pd.DataFrame(rows)
    out.insert(0, 'model', model.name)
    return out


def print_sanity(report: pd.DataFrame) -> None:
    """Readable print of ``sanity_report`` output."""
    marks = {PASS: '[ok]  ', WARN: '[warn]', FAIL: '[FAIL]', INFO: '[info]'}
    for model, g in report.groupby('model', sort=False):
        n_fail = int((g['status'] == FAIL).sum())
        n_warn = int((g['status'] == WARN).sum())
        print(f'\n=== {model}: {n_fail} fail, {n_warn} warn ===')
        for _, r in g.iterrows():
            print(f"{marks.get(r['status'], r['status'])} {r['check']}: {r['value']}  (expected {r['expected']})")
            if r['note'] and r['status'] in (WARN, FAIL):
                print(f"         {r['note']}")
