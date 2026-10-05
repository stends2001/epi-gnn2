"""
Component-recovery study: can the neural model (and hhh4 itself) recover the
endemic / epidemic / neighbourhood split when the truth is known?

1. Fit hhh4 in R to the real counts and simulate new count series from it, under
   scenarios that change the route of transmission (``r/hhh4_simulate.R``):
   ``fitted``, ``no_ne`` (no spread between regions), ``strong_ne`` (75% via
   neighbours). The true one-step components of every simulated week are known.
2. Train the neural model on each simulated series (lead 1, same graph, same
   date splits) and refit hhh4 on it.
3. Compare the estimated components on the test weeks with the truth:
   mu-weighted shares (all weeks, in season) and the per-region correlation of
   the shares.

This runs on simulated counts only, so it needs the R setup but not the data
pipeline beyond the real counts, graph and population used as a template.
"""
from __future__ import annotations

import copy
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .hhh4r import DEFAULT_SPEC, find_rscript, run_hhh4_r

R_SIM_SCRIPT = Path(__file__).resolve().parent / 'r' / 'hhh4_simulate.R'

MODULE_KEYS = {'hidden_size', 'num_layers', 'norm_edges', 'alpha_mode', 'endemic_mode', 'epidemic_mode',
               'neighbourhood_mode', 'node_effects', 'node_penalty', 'seasonal_rates', 'rate_dynamics',
               'dynamics_hidden', 'max_log_rate_adj', 'dynamics_penalty', 'disabled_branches'}


# --------------------------------------------------------------------------- #
# 1. simulate from a fitted hhh4
# --------------------------------------------------------------------------- #
def simulate_from_hhh4(counts: pd.DataFrame, adjacency: np.ndarray, population: np.ndarray,
                       fit_end: pd.Timestamp, scenarios: list[str], folder: str | Path,
                       spec: dict | None = None, rscript: str | None = None, sim_seed: int = 1,
                       timeout: int = 7200) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Returns {scenario: (simulated counts [dates x nodes], true components)}.
    True components: date, node, endemic, epidemic, neighbourhood, mean (one step).
    """
    exe = find_rscript(rscript)
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    spec = {**DEFAULT_SPEC, **(spec or {})}

    counts = counts.sort_index()
    dates = pd.to_datetime(counts.index)
    fit_end_row = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(fit_end)), side='right'))
    out = counts.copy()
    out.index = dates.strftime('%Y-%m-%d')
    out.columns = [f'node_{i}' for i in range(out.shape[1])]
    out.round().astype(int).to_csv(folder / 'counts.csv', index_label='date')
    pd.DataFrame((np.asarray(adjacency) > 0).astype(int)).to_csv(folder / 'adjacency.csv', header=False, index=False)
    pd.DataFrame({'node': range(len(population)), 'population': np.asarray(population, float)}) \
        .to_csv(folder / 'population.csv', index=False)
    kv = {**spec, 'fit_end_row': fit_end_row, 'scenarios': ' '.join(scenarios), 'sim_seed': sim_seed}
    pd.DataFrame({'key': list(kv), 'value': [str(v) for v in kv.values()]}).to_csv(folder / 'spec.csv', index=False)

    proc = subprocess.run([exe, str(R_SIM_SCRIPT), str(folder)], capture_output=True, text=True, timeout=timeout)
    (folder / 'r_log.txt').write_text(proc.stdout + '\n--- stderr ---\n' + proc.stderr)
    if proc.returncode != 0:
        tail = '\n'.join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f'hhh4 simulation in R failed (see {folder / "r_log.txt"}):\n{tail}')

    res = {}
    for sc in scenarios:
        f_counts, f_comp = folder / f'sim_{sc}_counts.csv', folder / f'sim_{sc}_components.csv'
        if not f_counts.exists():
            continue                                  # skipped in R (explosive)
        sim = pd.read_csv(f_counts, index_col=0)
        sim.index = pd.to_datetime(sim.index)
        sim.columns = range(sim.shape[1])
        comp = pd.read_csv(f_comp)
        comp['date'] = dates[(comp['row'] - 1).to_numpy()]
        res[sc] = (sim, comp.drop(columns='row'))
    return res


# --------------------------------------------------------------------------- #
# 2. the neural model on a plain count matrix
# --------------------------------------------------------------------------- #
def _snapshots(counts: pd.DataFrame, graph, seq_len: int, lead: int):
    from ..dataloading.databuilders.graphdatabuilder.datacontainers import Data
    y = counts.to_numpy(dtype=float)
    dates = pd.to_datetime(counts.index)
    week = dates.isocalendar().week.to_numpy().astype(float)
    n_weeks = np.where(pd.Series(dates.year).map(lambda yr: pd.Timestamp(year=yr, month=12, day=28)
                                                 .isocalendar()[1]).to_numpy() == 53, 53, 52)
    sinw, cosw = np.sin(2 * np.pi * week / n_weeks), np.cos(2 * np.pi * week / n_weeks)
    N = y.shape[1]
    snaps = []
    for t in range(seq_len - 1, len(y) - lead):
        x = np.stack([y[t - seq_len + 1:t + 1].T,
                      np.tile(sinw[t - seq_len + 1:t + 1], (N, 1)),
                      np.tile(cosw[t - seq_len + 1:t + 1], (N, 1))], axis=1)
        snaps.append((dates[t], dates[t + lead],
                      Data(torch.tensor(x, dtype=torch.float32),
                           torch.tensor(y[t + lead], dtype=torch.float32).view(N, 1), graph)))
    return snaps


def train_neural_on_counts(counts: pd.DataFrame, graph, splits: dict, model_cfg: dict, train_cfg: dict,
                           seq_len: int = 4, lead: int = 1, seed: int = 0, verbose: bool = False,
                           anchor: pd.DataFrame | None = None, anchor_weight: float = 1.0):
    """
    Train an HHH4Module on a count matrix (features: case window + week-of-year
    sin/cos, like the pipeline with ``target_column='cases'``). ``splits`` holds
    the date boundaries ``trainval`` and ``valtest`` (by target date). Returns
    the module and a function giving its components on the test targets.

    ``anchor``: one-step components of a fitted hhh4 (date, node, endemic,
    epidemic, neighbourhood) for the training weeks. The training loss then adds
    ``anchor_weight`` x the mean squared difference of the log components between
    the neural model and hhh4, so the neural model only moves the split away from
    hhh4 where that buys likelihood (lead 1 only, where both define the same
    components).
    """
    from ..models.gnnmodels.architectures.modules import HHH4Module
    from ..models.gnnmodels.utils import LossManager, Strategy

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    snaps = _snapshots(counts, graph, seq_len, lead)
    tv, vt = pd.Timestamp(splits['trainval']), pd.Timestamp(splits['valtest'])
    train = [s for _, td, s in snaps if td < tv]
    val   = [s for _, td, s in snaps if tv <= td < vt]
    test  = [(t0, td, s) for t0, td, s in snaps if td >= vt]
    if not train or not val or not test:
        raise ValueError('empty train / val / test split for the simulated series')

    ytr = torch.stack([s.y for s in train])
    node_means = ytr.mean(dim=(0, 2)).numpy()
    kw = {k: v for k, v in (model_cfg or {}).items() if k in MODULE_KEYS}
    if 'dropout' in (model_cfg or {}):
        kw['dropout_p'] = model_cfg['dropout']
    kw['disabled_branches'] = tuple(kw.get('disabled_branches') or ())
    N = counts.shape[1]
    m = HHH4Module(N, seq_len, 1, incidence_idx=[0], endemic_idx=[1, 2],
                   mu_init=float(node_means.mean()), node_means=node_means, **kw)

    opt = torch.optim.Adam(m.parameters(), lr=float(train_cfg.get('lr', 5e-3)))
    loss, strat = LossManager('nb'), Strategy()

    anchor_t = None
    if anchor is not None:
        if lead != 1:
            raise ValueError('anchoring to hhh4 components needs lead = 1')
        a = anchor.set_index(['date', 'node'])[['endemic', 'epidemic', 'neighbourhood']]
        eps = 0.05 * float(node_means.mean())
        anchor_t = {}
        for _, td, s in snaps:
            if td < tv and td in a.index.get_level_values(0):
                anchor_t[id(s)] = torch.log(torch.tensor(a.loc[td].reindex(range(N)).to_numpy(), dtype=torch.float32) + eps)
        log_eps = eps

    def train_step(s):
        if anchor_t is None or id(s) not in anchor_t:
            return strat.training_step(m, s, opt, loss)
        opt.zero_grad()
        (mu, alpha), parts = m(s.x, s.graph.edge_index, s.graph.edge_weight, return_components=True)
        l = loss((mu, alpha), s.y) + m.regularization()
        est = torch.log(torch.stack([parts[k][:, 0] for k in ('endemic', 'epidemic', 'neighbourhood')], 1) + log_eps)
        l = l + anchor_weight * ((est - anchor_t[id(s)]) ** 2).mean()
        l.backward()
        opt.step()
        return float(l.detach())
    best, best_state, wait = np.inf, None, 0
    patience = int(train_cfg.get('patience', 30))
    for epoch in range(int(train_cfg.get('n_epochs', 300))):
        m.train()
        for i in torch.randperm(len(train)).tolist():
            train_step(train[i])
        m.eval()
        v = float(np.mean([strat.validation_step(m, s, loss) for s in val]))
        if v < best - 1e-4:
            best, best_state, wait = v, copy.deepcopy(m.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and epoch % 10 == 0:
            print(f'   epoch {epoch}: val NB loss {v:.4f}')
    m.load_state_dict(best_state)
    m.eval()

    def components() -> pd.DataFrame:
        rows = []
        with torch.no_grad():
            for t0, td, s in test:
                (mu, a), parts = m(s.x, s.graph.edge_index, s.graph.edge_weight, return_components=True)
                rows.append(pd.DataFrame({'date': td, 'node': np.arange(N),
                                          **{k: v[:, 0].numpy() for k, v in parts.items()},
                                          'mean': mu[:, 0].numpy(), 'target': s.y[:, 0].numpy()}))
        return pd.concat(rows, ignore_index=True)

    return m, components


# --------------------------------------------------------------------------- #
# 3. compare with the truth
# --------------------------------------------------------------------------- #
def share_summary(comp: pd.DataFrame, season_weeks: tuple[int, int] | None = None) -> dict:
    """Mu-weighted shares over all rows (and in season if given)."""
    from ..models.utils.intervalmetrics import season_mask
    out = {}
    parts = [('all', comp)]
    if season_weeks:
        m = season_mask(comp['date'], *season_weeks).to_numpy()
        parts.append(('in_season', comp[m]))
    for tag, g in parts:
        tot = g[['endemic', 'epidemic', 'neighbourhood']].sum()
        tot = tot / max(tot.sum(), 1e-12)
        for k, v in tot.items():
            out[f'{tag}_share_{k}'] = float(v)
    return out


def node_share_correlation(est: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Per-node mu-weighted shares: Pearson correlation between estimate and truth."""
    def per_node(c):
        g = c.groupby('node')[['endemic', 'epidemic', 'neighbourhood']].sum()
        return g.div(g.sum(axis=1).clip(lower=1e-12), axis=0)
    e, t = per_node(est), per_node(truth).reindex(per_node(est).index)
    out = {}
    for k in ('endemic', 'epidemic', 'neighbourhood'):
        if e[k].std() > 0 and t[k].std() > 0:
            out[f'node_corr_{k}'] = float(np.corrcoef(e[k], t[k])[0, 1])
        else:
            out[f'node_corr_{k}'] = np.nan
    return out


def recovery_table(truth: pd.DataFrame, estimates: dict[str, pd.DataFrame],
                   season_weeks: tuple[int, int] | None = None) -> pd.DataFrame:
    """One row per estimator (plus 'truth'): shares and per-node correlations on the same rows."""
    rows = []
    keys = None
    for name, est in estimates.items():
        k = est[['date', 'node']].drop_duplicates()
        keys = k if keys is None else keys.merge(k, on=['date', 'node'])
    t = truth.merge(keys, on=['date', 'node'])
    rows.append({'estimator': 'truth', **share_summary(t, season_weeks)})
    for name, est in estimates.items():
        e = est.merge(keys, on=['date', 'node'])
        rows.append({'estimator': name, **share_summary(e, season_weeks), **node_share_correlation(e, t)})
    return pd.DataFrame(rows)
