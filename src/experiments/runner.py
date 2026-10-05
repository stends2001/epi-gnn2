"""
Experiment runner: turns a YAML config into tables, figures and a summary.

Every run writes to ``<output_dir>/<run name>/<timestamp>/``:

- ``config.yaml``   the fully resolved config (re-run it to reproduce)
- ``log.txt``       everything printed during the run
- ``summary.txt``   the headline numbers
- ``*.csv``         score, component and sanity tables
- ``figures/*.png`` all figures

The data / model steps are methods (``build_data``, ``graph_builder``,
``train_hhh4``, ``fit_baselines``) so tests can swap in synthetic versions.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_KEYS, TRAIN_KEYS, dump, epiconfig_kwargs, run_name


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


class Runner:
    def __init__(self, cfg: dict, out_root: str | Path | None = None, timestamp: bool = True):
        self.cfg = cfg
        root = Path(out_root or cfg.get('output_dir', 'results'))
        stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S') if timestamp else 'run'
        self.out = root / run_name(cfg) / stamp
        self.fig_dir = self.out / 'figures'
        self.summary: list[str] = []

    # ------------------------------------------------------------------ #
    # data and model steps (overridable)
    # ------------------------------------------------------------------ #
    def build_data(self, disease: str | None = None):
        from ..dataloading import EpiConfig, EpiDataOrchestrator, BaseLineDataBuilder
        epicfg = EpiConfig(**epiconfig_kwargs(self.cfg, disease))
        edo = EpiDataOrchestrator(epicfg).build()
        return edo, BaseLineDataBuilder(edo).build()

    def graph_builder(self, edo, kind: str = 'real', seed: int = 0):
        from ..dataloading import GraphDataBuilder
        from ..graphconstruction import identity_graph, rewired_graph
        gdb = GraphDataBuilder(edo).retrieve_static_graph(self.cfg['data']['graph_file'])
        if kind == 'identity':
            gdb.use_graph(identity_graph(gdb.graph.num_nodes))
        elif kind == 'rewired':
            gdb.use_graph(rewired_graph(gdb.graph, seed=seed))
        elif kind != 'real':
            raise ValueError(f'unknown graph kind {kind!r}')
        return gdb.build()

    def train_hhh4(self, gdb, name: str, seed: int):
        from ..models.gnnmodels import HHH4Model
        _set_seed(seed)
        m = HHH4Model(gdb, name=name)
        m.set_model_hparams(**{k: v for k, v in self.cfg.get('model', {}).items() if k in MODEL_KEYS})
        m.set_global_hparams(**{k: v for k, v in self.cfg.get('train', {}).items() if k in TRAIN_KEYS})
        m.train()
        if self.cfg.get('train', {}).get('calibrate_dispersion', False):
            m.dispersion_calibration = m.calibrate_dispersion('val', season=self.cfg.get('evaluation', {}).get('season', 'in'))
            print(f'   dispersion scale chosen on val: x{m.alpha_scale:.3g}')
        m.forecast('test')
        return m

    def fit_baselines(self, db) -> dict:
        from ..models import Persistence, SeasonalAverage
        b = self.cfg.get('baselines', {})
        scales = b.get('residual_scales', ['log1p'])
        out = {}
        for scale in scales:
            suffix = '' if len(scales) == 1 else f'_{scale}'
            out[f'persistence{suffix}'] = Persistence(
                db, name=f'persistence{suffix}', residual_scale=scale,
                min_bin_obs=b.get('min_bin_obs', 30))
            out[f'seasonal_average{suffix}'] = SeasonalAverage(
                db, name=f'seasonal_average{suffix}', residual_scale=scale,
                min_bin_obs=b.get('min_bin_obs', 30),
                train_residuals=b.get('train_residuals', 'leave_one_year_out'))
        for m in out.values():
            m.forecast('test')
        return out

    # ------------------------------------------------------------------ #
    def run(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(exist_ok=True)
        dump(self.cfg, self.out / 'config.yaml')

        import matplotlib
        matplotlib.use('Agg')

        t0 = time.time()
        with open(self.out / 'log.txt', 'w') as log, \
                contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            print(f"== {run_name(self.cfg)} ({self.cfg['task']}) -> {self.out}")
            getattr(self, f"task_{self.cfg['task']}")()
            print(f'\nFinished in {time.time() - t0:.0f} s')

        (self.out / 'summary.txt').write_text('\n'.join(self.summary) + '\n')
        print('\n'.join(['', 'SUMMARY'] + self.summary))
        print(f'\nResults in {self.out}')
        return self.out

    # ------------------------------------------------------------------ #
    # tasks
    # ------------------------------------------------------------------ #
    def task_baselines(self):
        from ..models.utils.intervalmetrics import compare_models
        from ..models import diagnostics as dg

        _, db = self.build_data()
        models = self.fit_baselines(db)
        ref = self._reference(models)
        for season in ('in', None):
            tab = compare_models(models, season=season, reference=ref)
            tag = 'in_season' if season else 'all_weeks'
            self._table(tab, f'scores_{tag}')
            if season == 'in':
                self._note_scores(tab, 'baselines, in season')
                self._fig(dg.plot_model_comparison(tab), 'model_comparison')

        for label, m in models.items():
            self._table(m.calibration_summary(), f'calibration_bins_{label}', show=False)
            self._sanity(m)
        self._fig(dg.plot_calibration(models), 'calibration')

    def task_hhh4(self):
        from ..models.utils.intervalmetrics import compare_models
        from ..models import diagnostics as dg

        ev = self.cfg.get('evaluation', {})
        edo, db = self.build_data()
        gdb = self.graph_builder(edo, 'real')
        baselines = self.fit_baselines(db)

        seeds = self.cfg.get('train', {}).get('seeds', [0])
        runs, per_seed = {}, []
        for seed in seeds:
            print(f'\n-- training seed {seed}')
            m = self.train_hhh4(gdb, name=f"hhh4_{self.cfg['data']['disease']}_s{seed}", seed=seed)
            runs[seed] = m
            tab = compare_models({'hhh4': m, **baselines}, season=ev.get('season', 'in'),
                                 reference=self._reference(baselines))
            ct = dg.component_table(m)
            ins = ct[ct['period'] == 'in-season'].iloc[0] if (ct['period'] == 'in-season').any() else ct.iloc[0]
            row = tab[tab['model'] == 'hhh4'].iloc[0]
            per_seed.append({'seed': seed, 'wis': row['wis'], 'rel_wis': row.get('rel_wis', np.nan),
                             'cov50': row.get('cov50'), 'cov80': row.get('cov80'), 'cov95': row.get('cov95'),
                             'share_endemic': ins['share_endemic'], 'share_epidemic': ins['share_epidemic'],
                             'share_neighbourhood': ins['share_neighbourhood']})
        per_seed = pd.DataFrame(per_seed)
        self._table(per_seed, 'hhh4_per_seed')
        if len(seeds) > 1:
            agg = per_seed.drop(columns='seed').agg(['mean', 'std'])
            self._table(agg, 'hhh4_over_seeds')
            self.summary.append(
                f"hhh4 over {len(seeds)} seeds: rel WIS {agg.loc['mean', 'rel_wis']:.3f} "
                f"(sd {agg.loc['std', 'rel_wis']:.3f}), neighbourhood share "
                f"{agg.loc['mean', 'share_neighbourhood']:.3f} (sd {agg.loc['std', 'share_neighbourhood']:.3f})")

        # full diagnostics on the first seed
        m = runs[seeds[0]]
        models = {'hhh4': m, **baselines}
        scores = compare_models(models, season=ev.get('season', 'in'), reference=self._reference(baselines))
        self._table(scores, 'scores_in_season')
        self._note_scores(scores, f'seed {seeds[0]}, in season')

        cal = getattr(m, 'dispersion_calibration', None)
        if cal is not None:
            self._table(cal, 'dispersion_calibration_val', show=False)
            best = cal.loc[cal['wis'].idxmin()]
            one = cal.iloc[(cal['alpha_scale'] - 1).abs().argmin()]
            self.summary.append(
                f"dispersion scale x{m.alpha_scale:.3g} (val in-season WIS {one['wis']:.3f} -> {best['wis']:.3f})")

        self._sanity(m, baselines)
        comp = dg.component_table(m)
        self._table(comp, 'components')
        self._table(dg.component_by_node(m), 'components_by_node', show=False)
        self._table(m.node_parameters(), 'node_parameters', show=False)
        ins = comp[comp['period'] == 'in-season']
        if len(ins):
            r = ins.iloc[0]
            self.summary.append(
                f"components in season (mu-weighted): endemic {r['share_endemic']:.2f}, "
                f"epidemic {r['share_epidemic']:.2f}, neighbourhood {r['share_neighbourhood']:.2f}; "
                f"bias {r['bias_ratio']:.2f}")

        if ev.get('figures', True):
            nodes = ev.get('nodes_to_plot', [0, 1, 2])
            self._fig(dg.plot_decomposition(m, nodes=nodes), 'decomposition')
            self._fig(dg.plot_component_shares(m), 'component_shares')
            with contextlib.suppress(ValueError):
                self._fig(dg.plot_rate_multipliers(m, nodes=nodes), 'rate_multipliers')
            self._fig(dg.plot_node_maps(m), 'node_maps')
            with contextlib.suppress(ValueError):
                self._fig(dg.plot_seasonal_curves(m), 'seasonal_curves')
            self._fig(dg.plot_calibration(models), 'calibration')
            self._fig(dg.plot_lag_check(models), 'lag_check')
            self._fig(dg.plot_pred_vs_obs(m), 'pred_vs_obs')
            self._fig(dg.plot_model_comparison(scores), 'model_comparison')

    def task_graph_controls(self):
        from ..models.utils.intervalmetrics import evaluate_model_intervals
        from ..models import diagnostics as dg
        import matplotlib.pyplot as plt

        gc = self.cfg.get('graph_controls', {})
        seeds = self.cfg.get('train', {}).get('seeds', [0])
        edo, _ = self.build_data()

        graphs = [('real', 0)]
        if gc.get('include_identity', True):
            graphs.append(('identity', 0))
        graphs += [('rewired', g) for g in range(int(gc.get('n_rewired', 10)))]

        rows = []
        for kind, gseed in graphs:
            gdb = self.graph_builder(edo, kind, seed=gseed)
            for seed in seeds:
                label = f'{kind}{gseed if kind == "rewired" else ""}_s{seed}'
                print(f'\n-- {label}')
                m = self.train_hhh4(gdb, name=label, seed=seed)
                ct = dg.component_table(m)
                ins = ct[ct['period'] == 'in-season'].iloc[0] if (ct['period'] == 'in-season').any() else ct.iloc[0]
                rows.append({'graph': kind, 'graph_draw': gseed, 'seed': seed,
                             'wis_in_season': evaluate_model_intervals(m, season='in')['wis'].iloc[0],
                             'share_endemic': ins['share_endemic'],
                             'share_epidemic': ins['share_epidemic'],
                             'share_neighbourhood': ins['share_neighbourhood']})
        res = pd.DataFrame(rows)
        self._table(res, 'runs', show=False)
        summ = res.groupby('graph')[['wis_in_season', 'share_neighbourhood']].agg(['mean', 'std', 'count'])
        self._table(summ, 'summary_by_graph')

        per_graph = res.groupby(['graph', 'graph_draw'])['wis_in_season'].mean()
        real = per_graph.loc[('real', 0)]
        if 'rewired' in per_graph.index.get_level_values(0):
            rew = per_graph.loc['rewired'].to_numpy()
            p = (1 + np.sum(rew <= real)) / (len(rew) + 1)
            seed_sd = res.loc[res['graph'] == 'real', 'wis_in_season'].std()
            self.summary += [
                f'real graph WIS {real:.4f}; rewired mean {rew.mean():.4f} '
                f'(range {rew.min():.4f}-{rew.max():.4f}, {len(rew)} graphs)',
                f'permutation p = {p:.3f} (smallest possible {1 / (len(rew) + 1):.3f}); '
                f'rewired - real = {rew.mean() - real:+.4f}, seed SD on real = '
                f'{seed_sd if not np.isnan(seed_sd) else 0:.4f}',
            ]
        if ('identity', 0) in per_graph.index:
            self.summary.append(f"identity graph WIS {per_graph.loc[('identity', 0)]:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        order = [g for g in ('real', 'rewired', 'identity') if g in set(res['graph'])]
        for ax, col, title in [(axes[0], 'wis_in_season', 'WIS in season (lower is better)'),
                               (axes[1], 'share_neighbourhood', 'Neighbourhood share in season')]:
            data = [res.loc[res['graph'] == g, col].to_numpy() for g in order]
            ax.boxplot(data, widths=0.5)
            ax.set_xticks(range(1, len(order) + 1), order)
            rng = np.random.default_rng(0)
            for i, d in enumerate(data):
                ax.scatter(np.full(len(d), i + 1) + rng.uniform(-0.12, 0.12, len(d)), d,
                           s=14, color=dg.plots.SERIES_COLORS[i], zorder=3)
            dg.plots._style(ax, title)
        fig.tight_layout()
        self._fig(fig, 'graph_controls')

    def task_compare_diseases(self):
        from ..models.utils.intervalmetrics import compare_models
        from ..models import diagnostics as dg

        diseases = self.cfg.get('compare', {}).get('diseases', ['norovirus', 'campylobacter', 'influenza'])
        seed = self.cfg.get('train', {}).get('seeds', [0])[0]
        models, scores = {}, []
        for disease in diseases:
            print(f'\n-- {disease}')
            edo, db = self.build_data(disease)
            m = self.train_hhh4(self.graph_builder(edo, 'real'), name=f'hhh4_{disease}', seed=seed)
            models[disease] = m
            baselines = self.fit_baselines(db)
            tab = compare_models({'hhh4': m, **baselines}, season='in', reference=self._reference(baselines))
            tab.insert(0, 'disease', disease)
            scores.append(tab)
            rep = dg.sanity_report(m, baselines=baselines)
            self._table(rep, f'sanity_{disease}', show=False)
            if self.cfg.get('evaluation', {}).get('figures', True):
                self._fig(dg.plot_component_shares(m), f'{disease}_component_shares')
                self._fig(dg.plot_node_maps(m), f'{disease}_node_maps')

        scores = pd.concat(scores, ignore_index=True)
        self._table(scores, 'scores_in_season')
        comp = dg.compare_components(models)
        self._table(comp, 'components_in_season')
        for _, r in comp.iterrows():
            rel = scores[(scores['disease'] == r['model']) & (scores['model'] == 'hhh4')]['rel_wis']
            self.summary.append(
                f"{r['model']:14s} rel WIS {rel.iloc[0] if len(rel) else float('nan'):.3f} | shares "
                f"endemic {r['share_endemic']:.2f}, epidemic {r['share_epidemic']:.2f}, "
                f"neighbourhood {r['share_neighbourhood']:.2f}")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _reference(self, models: dict) -> str | None:
        ref = self.cfg.get('evaluation', {}).get('reference', 'seasonal_average')
        if ref in models:
            return ref
        hits = [k for k in models if k.startswith(str(ref))]
        return hits[0] if hits else None

    def _table(self, df: pd.DataFrame, name: str, show: bool = True) -> None:
        df.to_csv(self.out / f'{name}.csv', index=not isinstance(df.index, pd.RangeIndex))
        if show:
            print(f'\n{name}:')
            with pd.option_context('display.width', 200, 'display.max_columns', 30):
                print(df.round(4).to_string())

    def _fig(self, fig, name: str) -> None:
        import matplotlib.pyplot as plt
        fig.savefig(self.fig_dir / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _sanity(self, model, baselines: dict | None = None) -> None:
        from ..models import diagnostics as dg
        rep = dg.sanity_report(model, baselines=baselines)
        self._table(rep, f'sanity_{model.name}', show=False)
        dg.print_sanity(rep)
        n_fail = int((rep['status'] == 'FAIL').sum())
        n_warn = int((rep['status'] == 'WARN').sum())
        self.summary.append(f'sanity {model.name}: {n_fail} fail, {n_warn} warn (see sanity_{model.name}.csv)')

    def _note_scores(self, tab: pd.DataFrame, what: str) -> None:
        cols = [c for c in ('wis', 'rel_wis', 'cov50', 'cov80', 'cov95') if c in tab.columns]
        self.summary.append(f'scores ({what}):')
        for _, r in tab.iterrows():
            self.summary.append('  ' + f"{r['model']:24s} " + '  '.join(f'{c}={r[c]:.3f}' for c in cols))


def _set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
