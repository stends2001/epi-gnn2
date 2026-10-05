"""
Attribution study on simulated data with a known split: does a ONE-week-ahead
model attribute cases to the right route better than a direct FOUR-week-ahead
model, and how do both compare with hhh4?

Data sources (all simulated, so the truth is known; no R needed):

- ``sir_c<c>``: model-neutral seasonal SIR waves on a grid, coupling c between
  neighbours (c = 0: no spread between regions at all);
- ``hhh4_<scenario>``: hhh4 (Python) fitted to an SIR series, then simulated with
  the route of transmission changed (fitted / no_ne / strong_ne).

Estimators:

- ``hhh4py``: hhh4 refitted on the simulated series;
- ``neural_lead1``: the neural model trained one week ahead (components are
  one-step transmission routes, like hhh4's);
- ``neural_lead<L>``: the neural model trained L weeks ahead directly (its
  components are L-week prediction weights).

Truth and estimates are compared on the test season (in season by default):
mu-weighted shares, their absolute error, and per-region share correlations. The
lead-L WIS of every estimator is reported too (hhh4py and neural_lead1 forecast
by simulating forward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .recovery import train_neural_on_counts, recovery_table, _snapshots
from .simulators import grid_adjacency, simulate_sir_network
from ..models.statistical import HHH4Py
from ..models.utils.intervalmetrics import wis, season_mask, pit_coverage, nb_randomized_pit

COMPS = ('endemic', 'epidemic', 'neighbourhood')


def _graph(adjacency):
    from ..graphconstruction import GraphStructure
    A = (np.asarray(adjacency) > 0).astype(int)
    i, j = np.nonzero(A)
    return GraphStructure(torch.tensor(np.stack([i, j]), dtype=torch.long),
                          torch.ones(len(i), dtype=torch.float32), A.shape[0])


def simulate_neural(module, counts: pd.DataFrame, graph, origins, lead: int, seq_len: int,
                    nsim: int = 200, seed: int = 0) -> np.ndarray:
    """
    Forward simulation with a ONE-step neural model: each step draws counts from
    NB(mu, alpha) and feeds them back. Returns draws [len(origins), nsim, N] at
    origin + lead. ``origins`` are row positions in ``counts``.
    """
    rng = np.random.default_rng(seed)
    y = counts.to_numpy(dtype=float)
    dates = pd.to_datetime(counts.index)
    week = dates.isocalendar().week.to_numpy().astype(float)
    n_weeks = np.where(pd.Series(dates.year).map(lambda yr: pd.Timestamp(year=yr, month=12, day=28)
                                                 .isocalendar()[1]).to_numpy() == 53, 53, 52)
    sinw, cosw = np.sin(2 * np.pi * week / n_weeks), np.cos(2 * np.pi * week / n_weeks)
    N = y.shape[1]
    ei, ew = graph.edge_index, graph.edge_weight
    out = np.empty((len(origins), nsim, N))
    module.eval()
    with torch.no_grad():
        for k, o in enumerate(origins):
            win = np.repeat(y[o - seq_len + 1:o + 1].T[None], nsim, axis=0)        # [nsim, N, S]
            for step in range(1, lead + 1):
                t_last = o + step - 1                                              # last week in window
                tw = np.arange(t_last - seq_len + 1, t_last + 1)
                feats = np.stack([np.tile(sinw[tw], (N, 1)), np.tile(cosw[tw], (N, 1))], axis=1)
                draws = np.empty((nsim, N))
                for s in range(nsim):
                    x = np.concatenate([win[s][:, None, :], feats], axis=1)
                    mu, alpha = module(torch.tensor(x, dtype=torch.float32), ei, ew)
                    mu, alpha = mu[:, 0].numpy(), alpha[:, 0].numpy()
                    draws[s] = rng.poisson(rng.gamma(1.0 / alpha, mu * alpha))
                win = np.concatenate([win[:, :, 1:], draws[:, :, None]], axis=2)
            out[k] = draws
    return out


def _score_draws(draws, y_true, quantiles, mask):
    q = np.quantile(draws, quantiles, axis=1, method='inverted_cdf')            # [Q, O, N]
    df = pd.DataFrame({'target': y_true.ravel(), **{f'pred_q{i+1}': q[i].ravel() for i in range(len(quantiles))}})
    df = df[mask.ravel()]
    yt = y_true.ravel()[mask.ravel()]
    d = draws.transpose(0, 2, 1).reshape(-1, draws.shape[1])[mask.ravel()]
    lo, hi = (d <= (yt[:, None] - 1)).mean(1), (d <= yt[:, None]).mean(1)
    pit = lo + np.random.default_rng(0).uniform(size=len(yt)) * (hi - lo)
    cov = pit_coverage(pit, quantiles)
    return float(wis(df, quantiles).mean()), dict(zip((cov['nominal'] * 100).round().astype(int), cov['coverage']))


def run_study(sources: list[str], replicates: list[int], model_cfg: dict, train_cfg: dict,
              side: int = 4, years: int = 9, lead: int = 4, seq_len: int = 4,
              quantiles=(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975), season_weeks=(44, 14),
              nsim: int = 200, template_coupling: float = 0.3, verbose: bool = True,
              estimators=('hhh4py', 'neural_lead1', 'neural_leadL', 'anchored'),
              anchor_weights=(1.0, 10.0)) -> pd.DataFrame:
    """Run all sources x replicates; returns one row per (source, replicate, estimator)."""
    quantiles = list(quantiles)
    A = grid_adjacency(side)
    graph = _graph(A)
    rows = []
    for rep in replicates:
        template, _, pop = simulate_sir_network(A, years, template_coupling, seed=1000 + rep)
        dates = template.index
        tv, vt = dates[(years - 2) * 52], dates[(years - 1) * 52]
        splits = {'trainval': tv, 'valtest': vt}
        fit_rows = np.arange(1, (years - 1) * 52)                 # targets before the test season
        hhh_template = None

        for src in sources:
            if src.startswith('sir_c'):
                counts, truth, popn = simulate_sir_network(A, years, float(src[5:]), seed=rep)
            elif src.startswith('hhh4_'):
                if hhh_template is None:
                    hhh_template = HHH4Py(harmonics=1, random_effects=True).fit(template.to_numpy(), A, pop,
                                                                                fit_rows=fit_rows)
                y, comp = hhh_template.simulate_series(src[5:], seed=rep)
                if y is None:
                    if verbose:
                        print(f'  {src} (rep {rep}): simulation exploded, skipped')
                    continue
                counts = pd.DataFrame(y, index=dates)
                comp['date'] = dates[comp['row'].to_numpy()]
                truth, popn = comp.drop(columns='row'), pop
            else:
                raise ValueError(f'unknown source {src!r}')

            test_rows = np.arange((years - 1) * 52, len(dates))
            estimates, scores = {}, {}

            # hhh4 refit (Python)
            h = HHH4Py(harmonics=1, random_effects=True).fit(counts.to_numpy(), A, popn, fit_rows=fit_rows)
            c = h.fitted_components(test_rows)
            c['date'] = dates[c['row'].to_numpy()]
            estimates['hhh4py'] = c.drop(columns='row')
            origins = np.arange((years - 1) * 52 - lead, len(dates) - lead)      # targets = test season
            target_dates = dates[origins + lead]
            y_true = counts.to_numpy()[origins + lead]
            mask = np.repeat(season_mask(pd.Series(target_dates), *season_weeks).to_numpy()[:, None], A.shape[0], 1)
            scores['hhh4py'] = _score_draws(h.simulate(origins, lead, nsim=nsim, seed=rep), y_true, quantiles, mask)

            # neural, one week ahead (+ simulation to lead L)
            if 'neural_lead1' in estimators:
                m1, comp1 = train_neural_on_counts(counts, graph, splits, model_cfg, train_cfg,
                                                   seq_len=seq_len, lead=1, seed=rep)
                estimates['neural_lead1'] = comp1()
                scores['neural_lead1'] = _score_draws(
                    simulate_neural(m1, counts, graph, origins, lead, seq_len, nsim=max(50, nsim // 4), seed=rep),
                    y_true, quantiles, mask)

            # neural, one week ahead, anchored to the hhh4 split
            if 'anchored' in estimators:
                train_rows = np.arange(1, (years - 2) * 52)
                anc = h.fitted_components(train_rows)
                anc['date'] = dates[anc['row'].to_numpy()]
                for kappa in anchor_weights:
                    label = f'anchored_k{kappa:g}'
                    ma, compa = train_neural_on_counts(counts, graph, splits, model_cfg, train_cfg, seq_len=seq_len,
                                                       lead=1, seed=rep, anchor=anc.drop(columns='row'),
                                                       anchor_weight=kappa)
                    estimates[label] = compa()
                    scores[label] = _score_draws(
                        simulate_neural(ma, counts, graph, origins, lead, seq_len, nsim=max(50, nsim // 4), seed=rep),
                        y_true, quantiles, mask)

            if 'neural_leadL' in estimators:
                # neural, L weeks ahead directly (NB forecasts for the same target weeks)
                from ..models.gnnmodels.architectures.modules import nb_quantiles
                mL, compL = train_neural_on_counts(counts, graph, splits, model_cfg, train_cfg,
                                                   seq_len=seq_len, lead=lead, seed=rep)
                estimates[f'neural_lead{lead}'] = compL()
                snaps = {td: s for t0, td, s in _snapshots(counts, graph, seq_len, lead)}
                mus, als = [], []
                with torch.no_grad():
                    for td in target_dates:
                        mu, a = mL(snaps[td].x, graph.edge_index, graph.edge_weight)
                        mus.append(mu[:, 0].numpy()); als.append(a[:, 0].numpy())
                mus, als = np.stack(mus), np.stack(als)
                qv = nb_quantiles(mus.ravel(), als.ravel(), quantiles)
                df = pd.DataFrame({'target': y_true.ravel(),
                                   **{f'pred_q{i+1}': qv[:, i] for i in range(len(quantiles))}})
                pit = nb_randomized_pit(y_true.ravel(), mus.ravel(), als.ravel())[mask.ravel()]
                cov = pit_coverage(pit, quantiles)
                scores[f'neural_lead{lead}'] = (float(wis(df[mask.ravel()], quantiles).mean()),
                                                dict(zip((cov['nominal'] * 100).round().astype(int), cov['coverage'])))

            tab = recovery_table(truth, estimates, season_weeks)
            tr = tab[tab['estimator'] == 'truth'].iloc[0]
            for _, r in tab.iterrows():
                row = {'source': src, 'replicate': rep, **r.to_dict()}
                if r['estimator'] != 'truth':
                    row['share_abs_error'] = float(sum(abs(r[f'in_season_share_{k}'] - tr[f'in_season_share_{k}'])
                                                       for k in COMPS) / 2)
                    w, cv = scores[r['estimator']]
                    row[f'wis_lead{lead}'] = w
                    row.update({f'pitcov{k}': v for k, v in cv.items()})
                rows.append(row)
            if verbose:
                show = tab[['estimator', 'in_season_share_endemic', 'in_season_share_epidemic',
                            'in_season_share_neighbourhood']].round(3)
                print(f'== {src} (rep {rep})\n{show.to_string(index=False)}')
                print('   lead-%d WIS: %s' % (lead, {k: round(v[0], 3) for k, v in scores.items()}), flush=True)
    return pd.DataFrame(rows)
