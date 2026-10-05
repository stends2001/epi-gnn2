"""
Figures for interval forecasts and the hhh4-style GNN.

Colours follow one fixed scheme across all figures:

- components: endemic = blue, epidemic = orange, neighbourhood = aqua;
- models in comparisons: categorical slots in a fixed order (never cycled);
- truth: dark ink; intervals: light grey bands;
- maps: one-hue sequential ramp for magnitudes, blue <-> red diverging ramp
  centred on 1 for multipliers.

Every function returns the matplotlib Figure, so it renders in notebooks and can
be saved with ``fig.savefig(...)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm

from ..utils.intervalmetrics import (
    model_prediction_frame, summarize_intervals, quantile_ranks, season_weeks_for, season_mask,
)

# ---- palette (validated: lightness, chroma, CVD separation) ----
COMPONENT_COLORS = {'endemic': '#2a78d6', 'epidemic': '#eb6834', 'neighbourhood': '#1baf7a'}
SERIES_COLORS    = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
WIS_PART_COLORS  = {'dispersion': '#4a3aa7', 'underprediction': '#eda100', 'overprediction': '#e87ba4'}
INK, INK_MUTED, BAND, GRID = '#1f1f1d', '#6b6a64', '#d9d8d3', '#e8e7e3'

SEQUENTIAL = LinearSegmentedColormap.from_list(
    'blue_seq', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_red', ['#184f95', '#6da7ec', '#f0efec', '#ef8a87', '#b3261e'])


def _style(ax, title: str | None = None, ylabel: str | None = None):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc='left')
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)


def _items(models):
    return list(models.items()) if isinstance(models, dict) else [(m.name, m) for m in models]


def _node_name(model, node: int) -> str:
    names = model.context_data.nodenames
    col = f'{model.epiconfig.level}_name'
    id_col = model.epiconfig.id_column
    if col in names.columns:
        hit = names.loc[names[id_col] == node, col]
        if len(hit):
            return f'{hit.iloc[0]} [node {node}]'
    return f'node {node}'


# --------------------------------------------------------------------------- #
# decomposition
# --------------------------------------------------------------------------- #
def plot_decomposition(model, nodes=(0,), dataset: str = 'test', horizon: int = 0,
                       show_interval: bool = True):
    """
    Per node: the forecast mean split into stacked endemic / epidemic /
    neighbourhood areas, the 95% and 50% prediction intervals, and the truth.
    The stacked areas add up to mu (the NB mean); the median is lower for skewed
    forecasts, which is why the truth can sit above the stack and still inside
    the interval.
    """
    from .components import components_with_target_time

    nodes = [nodes] if isinstance(nodes, (int, np.integer)) else list(nodes)
    comp  = components_with_target_time(model, dataset)
    comp  = comp[comp['horizon'] == horizon]
    preds = model_prediction_frame(model, dataset, horizon, True)
    t_col, id_col = model.epiconfig.temporal_column, model.epiconfig.id_column
    q = model.epiconfig.quantiles or []

    fig, axes = plt.subplots(len(nodes), 1, figsize=(12, 3.4 * len(nodes)), squeeze=False)
    for ax, node in zip(axes[:, 0], nodes):
        c = comp[comp[id_col] == node].sort_values('target_time')
        t = c['target_time']
        ax.stackplot(t, c['endemic'], c['epidemic'], c['neighbourhood'],
                     colors=[COMPONENT_COLORS[k] for k in ('endemic', 'epidemic', 'neighbourhood')],
                     labels=['endemic', 'epidemic', 'neighbourhood'], alpha=0.85,
                     edgecolor='white', linewidth=0.5)

        p = preds[preds[id_col] == node].sort_values(t_col)
        if show_interval and len(q) >= 3:
            n = len(q)
            ax.fill_between(p[t_col], p['pred_q1'], p[f'pred_q{n}'], color=BAND, alpha=0.6,
                            label=f'{int(round((1-2*q[0])*100))}% interval', zorder=0)
            if n >= 5:
                k = n // 2 - 1
                ax.plot(p[t_col], p[f'pred_q{k}'], color=INK_MUTED, lw=0.8, ls=':')
                ax.plot(p[t_col], p[f'pred_q{n-k+1}'], color=INK_MUTED, lw=0.8, ls=':',
                        label=f'{int(round((1-2*q[k-1])*100))}% interval')
        ax.plot(c['target_time'], c['target'], color=INK, lw=2, marker='o', ms=4, label='truth')
        _style(ax, _node_name(model, node), model.epiconfig.target_column)
        ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=3)
    fig.suptitle(f'{model.name}: forecast mean by component ({dataset})', x=0.01, ha='left',
                 color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def plot_component_shares(model, dataset: str = 'test', horizon: int = 0):
    """
    National view over time. Top: summed mu per branch (stacked) and the summed
    truth. Bottom: mu-weighted shares per week, which shows when in the season
    each branch matters.
    """
    from .components import components_with_target_time

    comp = components_with_target_time(model, dataset)
    comp = comp[comp['horizon'] == horizon]
    g = comp.groupby('target_time')[['endemic', 'epidemic', 'neighbourhood', 'mu', 'target']].sum()
    shares = g[['endemic', 'epidemic', 'neighbourhood']].div(g['mu'], axis=0)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 2]})
    cols = [COMPONENT_COLORS[k] for k in ('endemic', 'epidemic', 'neighbourhood')]
    a1.stackplot(g.index, g['endemic'], g['epidemic'], g['neighbourhood'], colors=cols,
                 labels=['endemic', 'epidemic', 'neighbourhood'], alpha=0.85,
                 edgecolor='white', linewidth=0.5)
    a1.plot(g.index, g['target'], color=INK, lw=2, label='truth (sum over regions)')
    _style(a1, 'Expected cases by component, summed over regions', model.epiconfig.target_column)
    a1.legend(loc='upper left', frameon=False, fontsize=9, ncol=4)

    a2.stackplot(shares.index, shares['endemic'], shares['epidemic'], shares['neighbourhood'],
                 colors=cols, alpha=0.85, edgecolor='white', linewidth=0.5)
    a2.set_ylim(0, 1)
    _style(a2, 'Share of the expected cases (mu-weighted)', 'share')
    fig.suptitle(model.name, x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
_MAP_SPECS = {
    'share_endemic':            ('Endemic share (in season)', 'seq', (0, 1)),
    'share_epidemic':           ('Epidemic share (in season)', 'seq', (0, 1)),
    'share_neighbourhood':      ('Neighbourhood share (in season)', 'seq', (0, 1)),
    'endemic_peak_week':        ('Endemic peak: weeks after the typical region', 'cyc', None),
    'seasonal_amplitude':       ('Seasonal peak / trough ratio', 'seq', None),
    'epidemic_rate':            ('Epidemic rate (cases per own case)', 'seq', None),
    'neighbourhood_rate':       ('Neighbourhood rate (per neighbour case)', 'seq', None),
    'epidemic_multiplier':      ('Epidemic node effect (x shared)', 'div', None),
    'neighbourhood_multiplier': ('Neighbourhood node effect (x shared)', 'div', None),
    'endemic_baseline':         ('Endemic baseline level', 'seq', None),
    'bias_ratio':               ('Bias: sum mu / sum truth', 'div', None),
    'alpha':                    ('NB dispersion', 'seq', None),
}


def plot_node_maps(model,
                   columns=('share_neighbourhood', 'endemic_peak_week', 'seasonal_amplitude',
                            'epidemic_rate', 'neighbourhood_rate', 'bias_ratio'),
                   dataset: str = 'test',
                   season: str | None = 'in'):
    """
    Choropleth maps of per-node results: component shares (in season), the
    endemic peak week and seasonal amplitude, the node effects, and the bias.
    Multipliers and the bias use a diverging scale centred on 1.
    """
    from .components import component_by_node

    table = component_by_node(model, dataset, season=season)
    shapes = model.context_data.local_shapedata
    id_col = model.epiconfig.id_column
    gdf = shapes.merge(table, on=id_col, how='left')

    columns = [c for c in columns if c in gdf.columns and gdf[c].notna().any()]
    ncol = 3 if len(columns) > 4 else max(len(columns), 1)
    nrow = int(np.ceil(len(columns) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5.6 * nrow), squeeze=False)

    for ax, col in zip(axes.ravel(), columns):
        title, kind, lim = _MAP_SPECS.get(col, (col, 'seq', None))
        vals = gdf[col].astype(float)
        if kind == 'cyc':
            # peak weeks wrap around new year: show the offset from the circular
            # mean peak, so week 52 and week 1 read as neighbours
            ang  = 2 * np.pi * vals / 52
            mean = np.angle(np.nanmean(np.exp(1j * ang))) * 52 / (2 * np.pi)
            off  = ((vals - mean + 26) % 52) - 26
            gdf['_offset'] = off
            span = max(np.nanmax(np.abs(off)), 1.0)
            norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
            gdf.plot(column='_offset', ax=ax, cmap=DIVERGING, norm=norm, legend=True,
                     edgecolor='white', linewidth=0.2, legend_kwds={'shrink': 0.6})
            title = f'{title} (typical: week {((mean - 1) % 52) + 1:.0f})'
        elif kind == 'div':
            # symmetric in log space, at least x1.25 either way, so small
            # deviations from 1 do not look dramatic
            r = np.nanmax(np.abs(np.log(vals.clip(lower=1e-9))))
            r = max(r, np.log(1.25))
            norm = TwoSlopeNorm(vmin=np.exp(-r), vcenter=1.0, vmax=np.exp(r))
            gdf.plot(column=col, ax=ax, cmap=DIVERGING, norm=norm, legend=True,
                     edgecolor='white', linewidth=0.2,
                     legend_kwds={'shrink': 0.6})
        else:
            vmin, vmax = lim if lim else (np.nanmin(vals), np.nanmax(vals))
            gdf.plot(column=col, ax=ax, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, legend=True,
                     edgecolor='white', linewidth=0.2, legend_kwds={'shrink': 0.6})
        ax.set_axis_off()
        ax.set_title(title, color=INK, fontsize=11, loc='left')
    for ax in axes.ravel()[len(columns):]:
        ax.set_axis_off()
    fig.suptitle(f'{model.name}: node parameters', x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def plot_seasonal_curves(model, nodes=None, horizon: int = 0, n_highlight: int = 5):
    """
    Endemic seasonal curves (relative to each node's own mean level) over the
    target week of year. Thin grey: all nodes; colour: ``nodes`` (default: the
    nodes with the largest amplitude); thick ink: the shared curve.
    """
    p = model.model.node_parameters()
    if 'endemic_coef' not in p:
        raise ValueError("needs endemic_mode='loglinear'")
    names = model.endemic_features
    si = next((i for i, f in enumerate(names) if f.endswith('sin_w')), None)
    ci = next((i for i, f in enumerate(names) if f.endswith('cos_w')), None)
    if si is None or ci is None:
        raise ValueError('needs week-of-year features (time_index_w=True) in the endemic branch')

    coef   = p['endemic_coef'][:, :, horizon]                   # [N, F]
    shared = model.model.end_coef.detach().cpu().numpy()[:, horizon]
    w      = np.arange(1, 53)
    th     = 2 * np.pi * w / 52
    lead   = model.epiconfig.horizon_leadtime + horizon
    tw     = ((w + lead - 1) % 52) + 1
    order  = np.argsort(tw)

    def curve(b, c):
        lr = b * np.sin(th) + c * np.cos(th)
        return np.exp(lr - lr.mean())

    curves = np.stack([curve(coef[i, si], coef[i, ci]) for i in range(coef.shape[0])])
    amp = curves.max(1) / curves.min(1)
    if nodes is None:
        nodes = list(np.argsort(-amp)[:n_highlight])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i in range(curves.shape[0]):
        ax.plot(tw[order], curves[i][order], color=BAND, lw=0.6, zorder=1)
    for k, node in enumerate(nodes):
        ax.plot(tw[order], curves[node][order], color=SERIES_COLORS[k % len(SERIES_COLORS)], lw=2,
                label=_node_name(model, int(node)), zorder=3)
    ax.plot(tw[order], curve(shared[si], shared[ci])[order], color=INK, lw=3, label='shared curve', zorder=4)
    ax.axhline(1, color=INK_MUTED, lw=0.8, ls='--')
    _style(ax, f'{model.name}: endemic seasonality per region (relative to own mean)',
           'endemic level / own mean')
    ax.set_xlabel('target week of year', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# calibration and skill
# --------------------------------------------------------------------------- #
def plot_calibration(models, dataset: str = 'test', season: str | None = 'in'):
    """
    Left: empirical vs nominal coverage per central interval (on the diagonal =
    calibrated; above = too wide; below = too narrow). Right: PIT histogram
    (share of truths per quantile-rank bin; flat = calibrated, U = too narrow,
    hump = too wide).
    """
    items = _items(models)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls='--')

    for k, (label, m) in enumerate(items):
        col = SERIES_COLORS[k % len(SERIES_COLORS)]
        q = m.epiconfig.quantiles
        df = model_prediction_frame(m, dataset, 0, True, season)
        s = summarize_intervals(df, q)
        a1.plot(s['nominal'], s['coverage'], color=col, lw=2, marker='o', ms=8, label=label)

        ranks = quantile_ranks(df, q)
        bins = np.linspace(0, 1, len(q) + 2)
        freq, _ = np.histogram(ranks, bins=bins)
        a2.step(np.arange(len(freq)), freq / freq.sum(), where='mid', color=col, lw=2, label=label)

        expected = np.diff(np.concatenate([[0], q, [1]]))
    a2.step(np.arange(len(expected)), expected, where='mid', color=INK, lw=1.2, ls='--', label='calibrated')

    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    _style(a1, f'Coverage ({season or "all"} season)', 'empirical coverage')
    a1.set_xlabel('nominal coverage', color=INK)
    a1.legend(frameon=False, fontsize=9)
    _style(a2, 'Where the truth falls among the quantiles', 'share of truths')
    a2.set_xlabel('number of predicted quantiles below the truth', color=INK)
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_lag_check(models, dataset: str = 'test'):
    """
    Correlation of the median forecast with the truth k weeks earlier. A curve
    that peaks at k = 0 is on time; one that peaks at the lead time copies the
    last observation.
    """
    from .sanity import lag_correlation

    items = _items(models)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lead = None
    for k, (label, m) in enumerate(items):
        lc = lag_correlation(m, dataset)
        ax.plot(lc['lag_weeks'], lc['corr'], color=SERIES_COLORS[k % len(SERIES_COLORS)],
                lw=2, marker='o', ms=8, label=label)
        lead = m.epiconfig.horizon_leadtime
    if lead is not None:
        ax.axvline(lead, color=INK_MUTED, lw=1, ls='--')
        ax.text(lead, ax.get_ylim()[0], ' lead time', color=INK_MUTED, va='bottom', fontsize=9)
    _style(ax, 'Lag check: median forecast vs truth k weeks earlier', 'correlation (log1p)')
    ax.set_xlabel('k (weeks)', color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_pred_vs_obs(model, dataset: str = 'test', season: str | None = 'in'):
    """Median forecast against truth (log1p axes), with the 1:1 line."""
    q = model.epiconfig.quantiles
    med = f'pred_q{len(q) // 2 + 1}' if q else model.epiconfig.pred_column
    df = model_prediction_frame(model, dataset, 0, True, season)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    x, y = np.log1p(df['target'].clip(lower=0)), np.log1p(df[med].clip(lower=0))
    hb = ax.hexbin(x, y, gridsize=40, cmap=SEQUENTIAL, mincnt=1, norm=LogNorm())
    fig.colorbar(hb, ax=ax, shrink=0.7, label='rows')
    lim = [0, max(x.max(), y.max()) * 1.02]
    ax.plot(lim, lim, color=INK, lw=1, ls='--')
    ax.set_xlim(lim); ax.set_ylim(lim)
    _style(ax, f'{model.name}: median vs truth ({season or "all"} season)', 'log1p(median forecast)')
    ax.set_xlabel('log1p(truth)', color=INK)
    fig.tight_layout()
    return fig


def plot_model_comparison(table: pd.DataFrame, horizon: int = 0):
    """
    WIS per model split into dispersion, underprediction and overprediction
    (left) and 50/80/95% coverage (right). Takes ``compare_models`` output.
    """
    t = table[table['horizon'] == horizon].sort_values('wis', ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.6 * len(t) + 2.6), sharey=True,
                                 gridspec_kw={'width_ratios': [3, 2]})
    left = np.zeros(len(t))
    for part, col in WIS_PART_COLORS.items():
        a1.barh(t['model'], t[part], left=left, color=col, label=part, height=0.6,
                edgecolor='white', linewidth=2)
        left += t[part].to_numpy()
    for y, v in enumerate(t['wis']):
        a1.text(v, y, f' {v:.3g}', va='center', color=INK, fontsize=9)
    _style(a1, 'WIS (lower is better)')
    a1.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)

    cov_cols = [c for c in t.columns if c.startswith('cov')]
    markers = ['o', 's', 'D', '^']
    for k, c in enumerate(sorted(cov_cols, key=lambda s: int(s[3:]))):
        nominal = int(c[3:]) / 100
        a2.scatter(t[c], t['model'], color=SERIES_COLORS[k], s=60, marker=markers[k % 4],
                   label=f'{int(nominal*100)}%', zorder=3)
        a2.axvline(nominal, color=SERIES_COLORS[k], lw=1, ls='--')
    a2.set_xlim(0, 1)
    _style(a2, 'Coverage (dashed = nominal)')
    a2.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    return fig
