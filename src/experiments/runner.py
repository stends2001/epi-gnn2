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
import copy
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
    def __init__(self, cfg: dict, out_root: str | Path | None = None, timestamp: bool = True,
                 out_dir: str | Path | None = None):
        self.cfg = cfg
        if out_dir is not None:
            self.out = Path(out_dir)
        else:
            root = Path(out_root or cfg.get('output_dir', 'results'))
            stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S') if timestamp else 'run'
            self.out = root / run_name(cfg) / stamp
        self.fig_dir = self.out / 'figures'
        self.summary: list[str] = []
        self.results: dict[str, pd.DataFrame] = {}     # every table written, by name

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
            seasons = self.cfg.get('data', {}).get('test_seasons')
            if seasons:
                self._run_seasons(list(seasons))
            else:
                getattr(self, f"task_{self.cfg['task']}")()
            print(f'\nFinished in {time.time() - t0:.0f} s')

        (self.out / 'summary.txt').write_text('\n'.join(self.summary) + '\n')
        print('\n'.join(['', 'SUMMARY'] + self.summary))
        print(f'\nResults in {self.out}')
        return self.out

    # ------------------------------------------------------------------ #
    # rolling test seasons
    # ------------------------------------------------------------------ #
    @staticmethod
    def season_dates(year: int) -> dict:
        """Test season = June of ``year`` to June of ``year + 1``; val = the season before."""
        return {'split_trainval': f'{year - 1}-06-01', 'split_valtest': f'{year}-06-01',
                'max_date': f'{year + 1}-06-01'}

    def _run_seasons(self, seasons: list[int]) -> None:
        """
        Run the task once per test season (expanding window: train on everything
        before the validation season), each in its own subfolder, then pool the
        tables over seasons. Every model is refitted per season.
        """
        per_season: dict[int, dict[str, pd.DataFrame]] = {}
        for year in seasons:
            print(f'\n#################### test season {year}/{str(year + 1)[-2:]} ####################')
            sub_cfg = copy.deepcopy(self.cfg)
            sub_cfg['data']['dates'] = {**sub_cfg['data']['dates'], **self.season_dates(year)}
            sub_cfg['data'].pop('test_seasons', None)
            sub = self.__class__(sub_cfg, out_dir=self.out / f'season_{year}')
            sub.out.mkdir(parents=True, exist_ok=True)
            sub.fig_dir.mkdir(exist_ok=True)
            dump(sub_cfg, sub.out / 'config.yaml')
            getattr(sub, f"task_{self.cfg['task']}")()
            (sub.out / 'summary.txt').write_text('\n'.join(sub.summary) + '\n')
            per_season[year] = sub.results
            self.summary.append(f'--- season {year}/{str(year + 1)[-2:]}')
            self.summary += ['  ' + line for line in sub.summary]

        self.summary.append(f'=== pooled over {len(seasons)} test seasons')
        names = set().union(*[set(r) for r in per_season.values()])
        for name in sorted(names):
            frames = []
            for year, res in per_season.items():
                if name in res:
                    df = res[name].reset_index() if not isinstance(res[name].index, pd.RangeIndex) else res[name].copy()
                    df.insert(0, 'season', year)
                    frames.append(df)
            pooled = pd.concat(frames, ignore_index=True)
            self._table(pooled, f'all_seasons_{name}', show=False)

        self._pool_scores()
        if self.cfg['task'] == 'graph_controls':
            self._pool_graph_controls()

    def _pool_scores(self) -> None:
        tab = self.results.get('all_seasons_scores_in_season')
        if tab is None:
            return
        keys = [k for k in ('disease', 'model') if k in tab.columns]
        cols = [c for c in ('wis', 'rel_wis', 'pitcov50', 'pitcov80', 'pitcov95', 'cov50', 'cov80', 'cov95')
                if c in tab.columns]
        agg = tab.groupby(keys)[cols].agg(['mean', 'std'])
        self._table(agg, 'pooled_scores_in_season')
        if 'rel_wis' in tab.columns:
            for key, g in tab.groupby(keys):
                key = key if isinstance(key, tuple) else (key,)
                wins = int((g['rel_wis'] < 1).sum())
                self.summary.append(
                    f"  {' / '.join(map(str, key)):32s} rel WIS {g['rel_wis'].mean():.3f} "
                    f"(sd {g['rel_wis'].std():.3f}), better than reference in {wins}/{len(g)} seasons")

    def _pool_graph_controls(self) -> None:
        runs = self.results.get('all_seasons_runs')
        if runs is None:
            return
        per_graph = runs.groupby(['graph', 'graph_draw'])['wis_in_season'].mean()
        if 'rewired' not in per_graph.index.get_level_values(0):
            return
        real = per_graph.loc[('real', 0)]
        rew = per_graph.loc['rewired'].to_numpy()
        p = (1 + np.sum(rew <= real)) / (len(rew) + 1)
        self.summary.append(
            f'  pooled: real WIS {real:.4f}, rewired mean {rew.mean():.4f}; permutation p = {p:.3f} '
            f'(graph draws averaged over seasons and seeds)')

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
            tab = compare_models({'neural_hhh4': m, **baselines}, season=ev.get('season', 'in'),
                                 reference=self._reference(baselines))
            ct = dg.component_table(m)
            ins = ct[ct['period'] == 'in-season'].iloc[0] if (ct['period'] == 'in-season').any() else ct.iloc[0]
            row = tab[tab['model'] == 'neural_hhh4'].iloc[0]
            per_seed.append({'seed': seed, 'wis': row['wis'], 'rel_wis': row.get('rel_wis', np.nan),
                             'pitcov50': row.get('pitcov50'), 'pitcov80': row.get('pitcov80'),
                             'pitcov95': row.get('pitcov95'),
                             'share_endemic': ins['share_endemic'], 'share_epidemic': ins['share_epidemic'],
                             'share_neighbourhood': ins['share_neighbourhood']})
        per_seed = pd.DataFrame(per_seed)
        self._table(per_seed, 'hhh4_per_seed')
        if len(seeds) > 1:
            agg = per_seed.drop(columns='seed').agg(['mean', 'std'])
            self._table(agg, 'hhh4_over_seeds')
            self.summary.append(
                f"neural_hhh4 over {len(seeds)} seeds: rel WIS {agg.loc['mean', 'rel_wis']:.3f} "
                f"(sd {agg.loc['std', 'rel_wis']:.3f}), neighbourhood share "
                f"{agg.loc['mean', 'share_neighbourhood']:.3f} (sd {agg.loc['std', 'share_neighbourhood']:.3f})")

        # full diagnostics on the first seed
        m = runs[seeds[0]]
        models = {'neural_hhh4': m}
        for label, extra in self._extra_neural_variants(gdb, seeds[0]).items():
            models[label] = extra
        refs = self._reference_models(edo, gdb, m, seed=seeds[0])
        models.update(refs)
        models.update(baselines)
        scores = compare_models(models, season=ev.get('season', 'in'), reference=self._reference(baselines))
        self._table(scores, 'scores_in_season')
        self._note_scores(scores, f'seed {seeds[0]}, in season')
        self._report_references(refs)

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
            self._fig(dg.plot_lag_check({k: v for k, v in models.items() if k not in refs}), 'lag_check')
            self._fig(dg.plot_pred_vs_obs(m), 'pred_vs_obs')
            self._fig(dg.plot_model_comparison(scores), 'model_comparison')

    # ---- reference models: hhh4 (Python, R) and the one-step neural model ----
    def fit_hhh4_py(self, edo, gdb, reference, tag: str = ''):
        """hhh4 in Python on the same data, graph and forecast rows as ``reference`` (overridable)."""
        from .hhh4r import pipeline_counts
        from .onestep import hhh4_py_model, reference_origins
        rc = {k: v for k, v in self.cfg.get('hhh4_py', {}).items() if k != 'enabled'}
        counts, pop = pipeline_counts(edo)
        lead = int(edo.config.horizon_leadtime)
        fit_end = pd.Timestamp(edo.data_context.temporal_summary.split_valtest) - pd.Timedelta(days=1)
        return hhh4_py_model(counts, gdb.graph.adjacency_matrix.cpu().numpy(), pop, fit_end,
                             reference_origins(reference, lead), lead, edo.config, spec=rc)

    def fit_neural_sim(self, edo, reference, seed: int, tag: str = ''):
        """
        The neural model trained ONE week ahead, forecasting ``data.lead`` weeks
        ahead by simulation (overridable). Its components are one-step routes,
        comparable with hhh4's.
        """
        from .hhh4r import pipeline_counts
        from .onestep import neural_sim_model, reference_origins
        lead = int(self.cfg['data']['lead'])
        saved = self.cfg['data']['lead']
        self.cfg['data']['lead'] = 1
        try:
            edo1, _ = self.build_data(getattr(edo.config, 'disease', None))
            m1 = self.train_hhh4(self.graph_builder(edo1, 'real'),
                                 name=f"onestep_{self.cfg['data']['disease']}_s{seed}{tag}", seed=seed)
        finally:
            self.cfg['data']['lead'] = saved
        counts, _ = pipeline_counts(edo1)
        return neural_sim_model(m1, counts, reference_origins(reference, lead), lead, edo.config,
                                nsim=int(self.cfg.get('train', {}).get('sim_nsim', 200)), seed=seed)

    def _reference_models(self, edo, gdb, reference, seed: int = 0, tag: str = '') -> dict:
        """hhh4_py, hhh4_R and neural_hhh4_sim, as enabled in the config; failures are noted, not fatal."""
        out = {}
        if self.cfg.get('hhh4_py', {}).get('enabled', True):
            print('\n-- fitting hhh4 in Python (reference model)')
            try:
                t = time.time()
                m = self.fit_hhh4_py(edo, gdb, reference, tag)
                out['hhh4_py'] = m
                print(f"   converged={m.info.get('converged')} loglik={m.info.get('loglik', float('nan')):.1f} "
                      f"({time.time() - t:.0f} s)")
            except Exception as e:
                print(f'   hhh4 in Python failed: {e}')
                self.summary.append(f'hhh4_py failed: {str(e).splitlines()[0]}')
        r_model = self._maybe_hhh4_r(edo, gdb, reference, tag)
        if r_model is not None:
            out['hhh4_R'] = r_model
        if self.cfg.get('train', {}).get('one_step', False):
            print('\n-- neural model trained one week ahead, forecasting by simulation')
            out['neural_hhh4_sim'] = self.fit_neural_sim(edo, reference, seed, tag)
        return out

    def fit_hhh4_r(self, edo, gdb, reference, tag: str = ''):
        """hhh4 fitted in R on the same data, graph and forecast rows (overridable)."""
        from .hhh4r import hhh4_r_from_pipeline
        rc = self.cfg.get('hhh4_r', {})
        spec = {k: v for k, v in rc.items() if k not in ('enabled', 'rscript')}
        return hhh4_r_from_pipeline(edo, gdb.graph, reference, self.out / f'hhh4_R{tag}', spec=spec,
                                    rscript=rc.get('rscript'))

    def _maybe_hhh4_r(self, edo, gdb, reference, tag: str = ''):
        rc = self.cfg.get('hhh4_r', {})
        if not rc.get('enabled', False):
            return None
        from .hhh4r import RNotAvailable
        print('\n-- fitting hhh4 in R (reference model)')
        try:
            r_model = self.fit_hhh4_r(edo, gdb, reference, tag)
        except RNotAvailable as e:
            print(f'   skipped: {e}')
            self.summary.append(f'hhh4_R skipped: {e}')
            return None
        except Exception as e:      # keep the rest of the run
            print(f'   hhh4 in R failed: {e}')
            self.summary.append(f'hhh4_R failed: {str(e).splitlines()[0]} (see hhh4_R/r_log.txt)')
            return None
        info = r_model.fit_info
        print(f"   converged={info.get('converged')} random_effects={info.get('random_effects')} "
              f"runtime={float(info.get('runtime_s', 0)):.0f}s")
        return r_model

    def _report_references(self, refs: dict) -> None:
        """Parameter and one-step component tables of the reference models."""
        for label, r in refs.items():
            for attr in ('coefficients', 'unit_effects'):
                if getattr(r, attr, None) is not None:
                    self._table(getattr(r, attr), f'{label}_{attr}', show=False)
            if getattr(r, 'components', None) is None:
                continue
            self._table(r.components, f'{label}_components_one_step', show=False)
            ct = r.component_table()
            self._table(ct, f'{label}_component_table')
            ins = ct[ct['period'] == 'in-season']
            if len(ins):
                x = ins.iloc[0]
                self.summary.append(
                    f"{label} one-step components in season: endemic {x['share_endemic']:.2f}, "
                    f"epidemic {x['share_epidemic']:.2f}, neighbourhood {x['share_neighbourhood']:.2f}")

    def _extra_neural_variants(self, gdb, seed: int) -> dict:
        """Optional comparison variants of the neural model (evaluation.variants)."""
        out = {}
        for label, overrides in (self.cfg.get('evaluation', {}).get('variants') or {}).items():
            print(f'\n-- variant {label}: {overrides}')
            out[label] = self.train_hhh4_variant(gdb, f'{label}_s{seed}', seed, overrides)
        return out

    def train_hhh4_variant(self, gdb, name: str, seed: int, model_overrides: dict):
        """train_hhh4 with some model hyperparameters changed (overridable)."""
        saved = self.cfg.get('model', {})
        self.cfg['model'] = {**saved, **(model_overrides or {})}
        try:
            return self.train_hhh4(gdb, name=name, seed=seed)
        finally:
            self.cfg['model'] = saved

    def task_ablations(self):
        """
        The model ladder and branch ablations, all on the same data and seeds:
        each variant changes the full model in one respect. Reports WIS in
        season relative to the full model (> 1: the removed part helped), and
        includes hhh4 in R when enabled.
        """
        from ..models.utils.intervalmetrics import compare_models
        from ..models import diagnostics as dg

        ev = self.cfg.get('evaluation', {})
        variants = self.cfg.get('ablations', {}).get('variants') or {}
        seeds = self.cfg.get('train', {}).get('seeds', [0])
        edo, db = self.build_data()
        gdb = self.graph_builder(edo, 'real')
        baselines = self.fit_baselines(db)

        rows, first = [], {}
        for seed in seeds:
            full = self.train_hhh4(gdb, name=f'full_s{seed}', seed=seed)
            models = {'full': full}
            for label, overrides in variants.items():
                print(f'\n-- {label} (seed {seed}): {overrides}')
                models[label] = self.train_hhh4_variant(gdb, f'{label}_s{seed}', seed, overrides)
            if seed == seeds[0]:
                first = dict(models)
                refs = self._reference_models(edo, gdb, full, seed=seed)
                models.update(refs)
                first.update(refs)
            tab = compare_models({**models, **baselines}, season=ev.get('season', 'in'), reference='full')
            tab.insert(0, 'seed', seed)
            rows.append(tab)
            for label, mm in models.items():
                if hasattr(mm, 'forecast_components'):
                    ct = dg.component_table(mm)
                    ins = ct[ct['period'] == 'in-season']
                    if len(ins):
                        r = ins.iloc[0]
                        idx = (tab['model'] == label)
                        for c in ('share_endemic', 'share_epidemic', 'share_neighbourhood'):
                            tab.loc[idx, c] = r[c]
        res = pd.concat(rows, ignore_index=True)
        self._table(res, 'ablation_runs', show=False)
        cols = [c for c in ('wis', 'rel_wis', 'pitcov50', 'pitcov95', 'share_endemic', 'share_epidemic',
                            'share_neighbourhood') if c in res.columns]
        summ = res.groupby('model')[cols].agg(['mean', 'std'])
        self._table(summ, 'ablation_summary')
        self._table(res[res['seed'] == seeds[0]].drop(columns='seed'), 'scores_in_season', show=False)
        self.summary.append('ablations (WIS relative to the full model; > 1 = the change hurts):')
        for label, g in res.groupby('model'):
            self.summary.append(f"  {label:26s} rel WIS {g['rel_wis'].mean():.3f} (sd {g['rel_wis'].std():.3f})"
                                if len(g) > 1 else f"  {label:26s} rel WIS {g['rel_wis'].mean():.3f}")
        if ev.get('figures', True):
            sc = res[res['seed'] == seeds[0]].drop(columns='seed')
            self._fig(dg.plot_model_comparison(sc), 'ablation_comparison')

    def recovery_inputs(self):
        """Counts, population and graph used as the simulation template (overridable)."""
        from .hhh4r import pipeline_counts
        edo, _ = self.build_data()
        graph = self.graph_builder(edo, 'real').graph
        counts, pop = pipeline_counts(edo)
        return counts, pop, graph

    def task_recovery(self):
        """
        Simulate count series from a fitted hhh4 under scenarios with a known
        endemic / epidemic / neighbourhood split, refit the neural model and hhh4
        on them, and compare the estimated split with the truth.
        """
        import matplotlib.pyplot as plt
        from .recovery import train_neural_on_counts, recovery_table
        from .onestep import fit_hhh4_py
        from ..models.utils.intervalmetrics import season_weeks_for
        from ..models import diagnostics as dg

        rc = self.cfg.get('recovery', {})
        h_spec = {k: v for k, v in self.cfg.get('hhh4_py', {}).items() if k != 'enabled'}
        scenarios = rc.get('scenarios', ['fitted', 'no_ne', 'strong_ne'])
        lead = int(rc.get('lead', 1))
        seq_len = int(self.cfg['data'].get('sequence_length', 4))
        seeds = self.cfg.get('train', {}).get('seeds', [0])
        anchors = [float(k) for k in (rc.get('anchor_weights') or [])]
        dates = self.cfg['data']['dates']
        splits = {'trainval': pd.Timestamp(dates['split_trainval']), 'valtest': pd.Timestamp(dates['split_valtest'])}
        season = season_weeks_for(self.cfg['data']['disease'])

        counts, pop, graph = self.recovery_inputs()
        counts = counts.sort_index()
        idx = pd.to_datetime(counts.index)
        adjacency = graph.adjacency_matrix.cpu().numpy()
        fit_end = splits['valtest'] - pd.Timedelta(days=1)
        print(f'\n-- fitting hhh4 (Python) to the real counts as simulation template')
        template = fit_hhh4_py(counts, adjacency, pop, fit_end, h_spec)
        n_test0 = int(np.searchsorted(idx.values, np.datetime64(splits['valtest'])))
        n_train = int(np.searchsorted(idx.values, np.datetime64(splits['trainval'])))
        tables = []
        for sc in scenarios:
            y, truth = template.simulate_series(sc, seed=int(rc.get('sim_seed', 1)))
            if y is None:
                print(f'-- {sc}: simulated process exploded, skipped')
                self.summary.append(f'recovery, scenario {sc}: simulation exploded, skipped')
                continue
            sim = pd.DataFrame(y, index=counts.index)
            truth['date'] = idx[truth['row'].to_numpy()]
            truth = truth.drop(columns='row')
            pd.DataFrame(y, index=counts.index).to_csv(self.out / f'simulated_{sc}.csv')
            estimates = {}
            print(f'-- {sc}: refitting hhh4')
            h = fit_hhh4_py(sim, adjacency, pop, fit_end, h_spec)
            c = h.fitted_components(np.arange(max(n_test0, 1), len(idx)))
            c['date'] = idx[c['row'].to_numpy()]
            estimates['hhh4_refit'] = c.drop(columns='row')
            for seed in seeds:
                print(f'\n-- {sc}: neural model, seed {seed}')
                _, comp = train_neural_on_counts(sim, graph, splits, self.cfg.get('model', {}),
                                                 self.cfg.get('train', {}), seq_len=seq_len, lead=lead, seed=seed)
                estimates[f'neural_s{seed}'] = comp()
                if anchors and lead == 1:
                    anc = h.fitted_components(np.arange(1, n_train))
                    anc['date'] = idx[anc['row'].to_numpy()]
                    for k in anchors:
                        print(f'-- {sc}: neural model anchored on hhh4 (weight {k:g}), seed {seed}')
                        _, comp = train_neural_on_counts(sim, graph, splits, self.cfg.get('model', {}),
                                                         self.cfg.get('train', {}), seq_len=seq_len, lead=1,
                                                         seed=seed, anchor=anc.drop(columns='row'), anchor_weight=k)
                        estimates[f'anchored_k{k:g}_s{seed}'] = comp()
            tab = recovery_table(truth, estimates, season)
            tab.insert(0, 'scenario', sc)
            tables.append(tab)
        if not tables:
            return
        res = pd.concat(tables, ignore_index=True)
        self._table(res, 'recovery')

        for sc, g in res.groupby('scenario', sort=False):
            tr = g[g['estimator'] == 'truth'].iloc[0]
            self.summary.append(f'recovery, scenario {sc} (in-season neighbourhood share; truth {tr["in_season_share_neighbourhood"]:.2f}):')
            for _, r in g[g['estimator'] != 'truth'].iterrows():
                self.summary.append(
                    f"  {r['estimator']:14s} {r['in_season_share_neighbourhood']:.2f}  "
                    f"(endemic {r['in_season_share_endemic']:.2f} vs {tr['in_season_share_endemic']:.2f}; "
                    f"per-region corr of neighbourhood share {r.get('node_corr_neighbourhood', float('nan')):.2f})")

        # figure: in-season shares per scenario, truth vs estimators
        comps = ('endemic', 'epidemic', 'neighbourhood')
        scs = list(dict.fromkeys(res['scenario']))
        fig, axes = plt.subplots(1, len(scs), figsize=(4.5 * len(scs), 4), squeeze=False, sharey=True)
        for ax, sc in zip(axes[0], scs):
            g = res[res['scenario'] == sc]
            names = list(g['estimator'])
            bottom = np.zeros(len(names))
            for k in comps:
                vals = g[f'in_season_share_{k}'].to_numpy()
                ax.bar(names, vals, bottom=bottom, color=dg.plots.COMPONENT_COLORS[k], label=k,
                       edgecolor='white', linewidth=2, width=0.6)
                bottom += vals
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=30)
            dg.plots._style(ax, f'scenario: {sc}', 'share of expected cases (in season)')
        axes[0, 0].legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        fig.tight_layout()
        self._fig(fig, 'recovery_shares')

    def task_attribution(self):
        """
        Simulation study, no surveillance data needed: on series with a KNOWN
        endemic / epidemic / neighbourhood split (model-neutral SIR waves on a
        grid, and hhh4-generated series), does a model trained one week ahead
        attribute cases to the right route better than one trained L weeks ahead
        directly, and how do both compare with hhh4? Also scores the lead-L
        forecasts of every estimator.
        """
        import matplotlib.pyplot as plt
        from .attribution import run_study, COMPS
        from ..models import diagnostics as dg

        ac = self.cfg.get('attribution', {})
        lead = int(ac.get('lead', 4))
        res = run_study(list(ac.get('sources', ['sir_c0.0', 'sir_c0.3', 'sir_c0.6', 'hhh4_fitted', 'hhh4_no_ne',
                                                'hhh4_strong_ne'])),
                        list(ac.get('replicates', [0, 1, 2])), self.cfg.get('model', {}), self.cfg.get('train', {}),
                        side=int(ac.get('side', 4)), years=int(ac.get('years', 9)), lead=lead,
                        seq_len=int(self.cfg['data'].get('sequence_length', 4)),
                        quantiles=list(self.cfg['data']['quantiles']), nsim=int(ac.get('nsim', 200)),
                        template_coupling=float(ac.get('template_coupling', 0.3)),
                        estimators=tuple(ac.get('estimators', ['hhh4py', 'neural_lead1', 'neural_leadL'])),
                        anchor_weights=tuple(ac.get('anchor_weights', [1.0, 10.0])))
        self._table(res, 'attribution_runs', show=False)
        est = res[res['estimator'] != 'truth']
        cols = ['share_abs_error', f'wis_lead{lead}'] + [f'in_season_share_{k}' for k in COMPS]
        summ = est.groupby(['source', 'estimator'])[cols].mean().reset_index()
        truth = res[res['estimator'] == 'truth'].groupby('source')[[f'in_season_share_{k}' for k in COMPS]].mean()
        self._table(summ, 'attribution_summary')
        self._table(truth.reset_index(), 'attribution_truth', show=False)
        overall = est.groupby('estimator')[['share_abs_error', f'wis_lead{lead}']].mean()
        self._table(overall.reset_index(), 'attribution_overall')
        self.summary.append('attribution error (half the L1 distance of in-season shares to the truth; 0 = exact):')
        for e, r in overall.sort_values('share_abs_error').iterrows():
            self.summary.append(f"  {e:16s} share error {r['share_abs_error']:.3f}   lead-{lead} WIS {r[f'wis_lead{lead}']:.3f}")

        sources = list(dict.fromkeys(summ['source']))
        names = list(dict.fromkeys(summ['estimator']))
        fig, ax = plt.subplots(figsize=(1.6 * len(sources) + 3, 4))
        w = 0.8 / max(len(names), 1)
        for i, n in enumerate(names):
            g = summ[summ['estimator'] == n].set_index('source').reindex(sources)
            ax.bar(np.arange(len(sources)) + i * w, g['share_abs_error'], width=w, label=n,
                   color=dg.plots.SERIES_COLORS[i % len(dg.plots.SERIES_COLORS)])
        ax.set_xticks(np.arange(len(sources)) + 0.4 - w / 2, sources, rotation=20)
        ax.legend(frameon=False, fontsize=9)
        dg.plots._style(ax, 'attribution error by data source', 'share error (0 = exact)')
        fig.tight_layout()
        self._fig(fig, 'attribution_error')

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
            extra = self._reference_models(edo, self.graph_builder(edo, 'real'), m, seed=seed, tag=f'_{disease}')
            tab = compare_models({'neural_hhh4': m, **extra, **baselines}, season='in',
                                 reference=self._reference(baselines))
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
            rel = scores[(scores['disease'] == r['model']) & (scores['model'] == 'neural_hhh4')]['rel_wis']
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
        self.results[name] = df
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
        cols = [c for c in ('wis', 'rel_wis', 'pitcov50', 'pitcov80', 'pitcov95', 'cov50', 'cov80', 'cov95')
                if c in tab.columns]
        self.summary.append(f'scores ({what}):')
        for _, r in tab.iterrows():
            self.summary.append('  ' + f"{r['model']:24s} " + '  '.join(
                f'{c}={r[c]:.3f}' for c in cols if pd.notna(r[c])))


def _set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
