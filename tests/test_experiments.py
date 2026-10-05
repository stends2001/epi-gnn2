"""
Tests for the YAML configs, the run.py CLI and the experiment runner. The runner
is exercised end to end with synthetic models (no surveillance data needed).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from src.experiments.config import (
    ConfigError, load_config, set_dotted, deep_merge, run_name, epiconfig_kwargs,
)
from src.experiments.runner import Runner
import run as run_cli

CONFIGS = ROOT / 'configs'


# --------------------------------------------------------------------------- #
# config loading
# --------------------------------------------------------------------------- #
def test_all_shipped_configs_resolve():
    for p in CONFIGS.glob('*.yaml'):
        cfg = load_config(p, ['data.graph_file=grid.pt'])
        from src.experiments.config import TASKS
        assert cfg['task'] in TASKS
        assert run_name(cfg)


def test_extends_chain_and_overrides():
    cfg = load_config(CONFIGS / 'hhh4_campylobacter.yaml',
                      ['data.graph_file=g.pt', 'train.seeds=[3,4]', 'model.alpha_mode=global', 'data.lead=2'])
    assert cfg['data']['disease'] == 'campylobacter'          # own file
    assert cfg['train']['seeds'] == [3, 4]                    # override
    assert cfg['train']['lr'] == 0.005                        # from base, two levels up
    assert cfg['model']['alpha_mode'] == 'global'
    assert run_name(cfg) == 'hhh4_campylobacter_nuts3_lead2'
    kw = epiconfig_kwargs(cfg)
    assert kw['target_column'] == 'cases' and kw['lag_column'] == 'cases' and kw['horizon_leadtime'] == 2


def test_validation_messages():
    with pytest.raises(ConfigError, match='graph_file'):
        load_config(CONFIGS / 'hhh4_norovirus.yaml')
    with pytest.raises(ConfigError, match='unknown model keys'):
        load_config(CONFIGS / 'hhh4_norovirus.yaml', ['data.graph_file=g', 'model.hiden_size=3'])
    with pytest.raises(ConfigError, match='task must be'):
        load_config(CONFIGS / 'hhh4_norovirus.yaml', ['data.graph_file=g', 'task=train'])
    # baselines do not need a graph
    assert load_config(CONFIGS / 'baselines_norovirus.yaml')['task'] == 'baselines'


def test_merge_and_dotted():
    assert deep_merge({'a': {'b': 1, 'c': [1]}}, {'a': {'c': [2]}}) == {'a': {'b': 1, 'c': [2]}}
    d = {}
    set_dotted(d, 'x.y.z=[1, 2]')
    assert d == {'x': {'y': {'z': [1, 2]}}}
    with pytest.raises(ConfigError):
        set_dotted(d, 'novalue')


def test_cli_list_and_dry_run(capsys):
    assert run_cli.main(['--list']) == 0
    assert 'hhh4_norovirus.yaml' in capsys.readouterr().out
    assert run_cli.main(['configs/hhh4_norovirus.yaml', '--dry-run', '--set', 'data.disease=influenza']) == 0
    out = capsys.readouterr().out
    assert 'hhh4_influenza_nuts3_lead4' in out and 'graph_file is not set' in out
    assert run_cli.main(['configs/hhh4_norovirus.yaml']) == 2     # invalid: no graph file


# --------------------------------------------------------------------------- #
# runner, end to end on synthetic models
# --------------------------------------------------------------------------- #
class _SyntheticRunner(Runner):
    """Runner whose data/model steps return the synthetic models of test_diagnostics."""
    def build_data(self, disease=None):
        return SimpleNamespace(disease=disease), None

    def graph_builder(self, edo, kind='real', seed=0):
        return SimpleNamespace(kind=kind, seed=seed)

    def train_hhh4(self, gdb, name, seed):
        import test_diagnostics as T
        coupling = 0.0 if getattr(gdb, 'kind', 'real') == 'identity' else 0.3
        m, _, _ = T._build(coupling=coupling, epochs=8)
        m.name = name
        return m

    def fit_baselines(self, db):
        import test_diagnostics as T
        _, base, _ = T._build(coupling=0.0, epochs=1)
        seasonal = SimpleNamespace(**vars(base))
        seasonal.name = 'seasonal_average'
        for b in (base, seasonal):
            b.calibration_summary = lambda: pd.DataFrame({'seasonal_index': [1], 'n_obs': [40],
                                                         'horizon': [0], 'fallback': [False]})
        return {'persistence': base, 'seasonal_average': seasonal}


    def fit_hhh4_r(self, edo, gdb, reference, tag=''):
        """Real R fit on the synthetic counts (skipped if R is missing)."""
        import numpy as np
        import test_diagnostics as T
        from src.experiments.hhh4r import run_hhh4_r, HHH4RModel, find_rscript
        find_rscript()                                     # raises RNotAvailable without R
        t0, week, y = T._simulate(coupling=0.3)
        counts = pd.DataFrame(y, index=t0)
        ref = reference.predictions.get_preds('test').get(0, True, False)
        lead = reference.epiconfig.horizon_leadtime
        origins = sorted(pd.to_datetime(ref['timestamp']).unique() - pd.Timedelta(weeks=lead))
        res = run_hhh4_r(counts, T._grid_graph().adjacency_matrix.numpy(), np.full(y.shape[1], 1e5),
                         fit_end=origins[0], t0_dates=origins, lead=lead,
                         quantiles=reference.epiconfig.quantiles, folder=self.out / f'hhh4_R{tag}',
                         spec={'nsim': 50, 'random_effects': False})
        return HHH4RModel(res, reference.epiconfig)

    def recovery_inputs(self):
        import numpy as np
        import test_diagnostics as T
        t0, week, y = T._simulate(coupling=0.3)
        return pd.DataFrame(y, index=t0), np.full(y.shape[1], 1e5), T._grid_graph()


def _r_available():
    import shutil
    return shutil.which('Rscript') is not None


def _cfg(task, **extra):
    cfg = load_config(CONFIGS / 'base.yaml', ['data.graph_file=g.pt', f'task={task}',
                                             'train.seeds=[0]', *extra.get('sets', [])])
    return cfg


@pytest.mark.parametrize('task, sets, expect', [
    ('ablations', ['ablations.variants={no_gru: {rate_dynamics: none}, no_neighbourhood: {disabled_branches: [neighbourhood]}}',
                   'hhh4_r.enabled=false'],
     ['ablation_runs.csv', 'ablation_summary.csv', 'figures/ablation_comparison.png']),
    ('hhh4', ['train.seeds=[0,1]'],
     ['scores_in_season.csv', 'components.csv', 'node_parameters.csv', 'hhh4_over_seeds.csv',
      'figures/decomposition.png', 'figures/node_maps.png']),
    ('baselines', [], ['scores_in_season.csv', 'scores_all_weeks.csv', 'figures/calibration.png']),
    ('graph_controls', ['graph_controls.n_rewired=2'],
     ['runs.csv', 'summary_by_graph.csv', 'figures/graph_controls.png']),
    ('compare_diseases', ['compare.diseases=[norovirus,campylobacter]', 'evaluation.figures=false'],
     ['scores_in_season.csv', 'components_in_season.csv']),
])
def test_runner_tasks_write_outputs(tmp_path, task, sets, expect):
    cfg = _cfg(task, sets=sets)
    out = _SyntheticRunner(cfg, out_root=tmp_path, timestamp=False).run()
    for f in ['config.yaml', 'log.txt', 'summary.txt'] + expect:
        assert (out / f).exists(), f
    assert (out / 'summary.txt').read_text().strip()
    # the saved config re-loads and reproduces the run name
    again = load_config(out / 'config.yaml')
    assert run_name(again) == run_name(cfg)


@pytest.mark.skipif(not _r_available(), reason='R not installed')
def test_hhh4_r_reference_in_hhh4_task(tmp_path):
    cfg = _cfg('hhh4')
    out = _SyntheticRunner(cfg, out_root=tmp_path, timestamp=False).run()
    scores = pd.read_csv(out / 'scores_in_season.csv')
    assert 'hhh4_R' in set(scores['model'])
    row = scores[scores['model'] == 'hhh4_R'].iloc[0]
    assert 0 < row['pitcov95'] <= 1 and row['wis'] > 0
    assert (out / 'hhh4_R_coefficients.csv').exists() and (out / 'hhh4_R_component_table.csv').exists()


@pytest.mark.skipif(not _r_available(), reason='R not installed')
def test_recovery_task(tmp_path):
    cfg = _cfg('recovery', sets=['recovery.scenarios=[fitted, no_ne]', 'train.n_epochs=15', 'train.patience=5',
                                 'data.dates.split_trainval=2016-09-01', 'data.dates.split_valtest=2017-03-01',
                                 'hhh4_r.random_effects=false'])
    out = _SyntheticRunner(cfg, out_root=tmp_path, timestamp=False).run()
    rec = pd.read_csv(out / 'recovery.csv')
    assert set(rec['scenario']) == {'fitted', 'no_ne'}
    truth = rec[(rec['scenario'] == 'no_ne') & (rec['estimator'] == 'truth')].iloc[0]
    assert truth['all_share_neighbourhood'] == pytest.approx(0, abs=1e-6)    # no spread simulated
    assert {'neural_s0', 'hhh4_refit'} <= set(rec['estimator'])
    assert (out / 'figures' / 'recovery_shares.png').exists()


def test_rolling_seasons_pool_scores(tmp_path):
    cfg = _cfg('hhh4', sets=['data.test_seasons=[2016, 2017]', 'hhh4_r.enabled=false', 'evaluation.figures=false'])
    out = _SyntheticRunner(cfg, out_root=tmp_path, timestamp=False).run()
    pooled = pd.read_csv(out / 'all_seasons_scores_in_season.csv')
    assert set(pooled['season']) == {2016, 2017}
    assert (out / 'pooled_scores_in_season.csv').exists()
    assert (out / 'season_2016' / 'summary.txt').exists()
    assert 'pooled over 2 test seasons' in (out / 'summary.txt').read_text()
