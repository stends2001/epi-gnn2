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
        assert cfg['task'] in ('baselines', 'hhh4', 'graph_controls', 'compare_diseases')
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


def _cfg(task, **extra):
    cfg = load_config(CONFIGS / 'base.yaml', ['data.graph_file=g.pt', f'task={task}',
                                             'train.seeds=[0]', *extra.get('sets', [])])
    return cfg


@pytest.mark.parametrize('task, sets, expect', [
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
