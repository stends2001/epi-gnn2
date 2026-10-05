"""
YAML run configs.

A config file may ``extends:`` another file (path relative to itself); the child
is deep-merged over the parent, so an experiment file only lists what differs
from ``configs/base.yaml``. Command-line overrides use dotted keys::

    data.disease=campylobacter   train.seeds=[0,1,2]   model.alpha_mode=global

Values are parsed as YAML, so numbers, lists and booleans work as written.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

TASKS = ('baselines', 'hhh4', 'graph_controls', 'compare_diseases', 'ablations', 'recovery', 'attribution')

# keys passed straight to HHH4Model.set_model_hparams / set_global_hparams
MODEL_KEYS = {'hidden_size', 'num_layers', 'dropout', 'norm_edges', 'alpha_mode',
              'incidence_features', 'endemic_features', 'init_from_train', 'endemic_mode',
              'node_effects', 'node_penalty', 'neighbourhood_mode', 'epidemic_mode',
              'seasonal_rates', 'rate_dynamics', 'dynamics_hidden', 'max_log_rate_adj',
              'dynamics_penalty', 'disabled_branches'}
TRAIN_KEYS = {'lr', 'n_epochs', 'patience', 'min_delta', 'optimizer', 'scheduler',
              'optimizer_kwargs', 'scheduler_kwargs', 'shuffle'}
# train keys handled by the runner itself, not passed to set_global_hparams
RUNNER_TRAIN_KEYS = {'seeds', 'calibrate_dispersion', 'one_step', 'sim_nsim'}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge: dicts merge, everything else (incl. lists) is replaced."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None,
                check: bool = True) -> dict:
    """Load a YAML config, resolve ``extends`` chains, apply dotted overrides."""
    cfg = _load_with_extends(Path(path).resolve(), seen=set())
    for item in overrides or []:
        set_dotted(cfg, item)
    cfg['_source'] = str(Path(path))
    if check:
        validate(cfg)
    return cfg


def _load_with_extends(path: Path, seen: set) -> dict:
    if path in seen:
        raise ConfigError(f'circular extends: {path}')
    seen.add(path)
    if not path.exists():
        raise ConfigError(f'config not found: {path}')
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop('extends', None)
    if parent:
        base = _load_with_extends((path.parent / parent).resolve(), seen)
        cfg = deep_merge(base, cfg)
    return cfg


def set_dotted(cfg: dict, item: str) -> None:
    """Apply one ``a.b.c=value`` override in place (value parsed as YAML)."""
    if '=' not in item:
        raise ConfigError(f"override must look like key.sub=value, got {item!r}")
    key, raw = item.split('=', 1)
    value = yaml.safe_load(raw)
    parts = key.strip().split('.')
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f'cannot set {key}: {p} is not a section')
    node[parts[-1]] = value


def validate(cfg: dict) -> None:
    """Catch the common mistakes before any data is loaded."""
    errors = []
    task = cfg.get('task')
    if task not in TASKS:
        errors.append(f'task must be one of {TASKS}, got {task!r}')

    data = cfg.get('data', {})
    for k in ('disease', 'level', 'lead', 'quantiles', 'dates'):
        if k not in data:
            errors.append(f'data.{k} is missing')
    if task in ('hhh4', 'graph_controls', 'compare_diseases', 'ablations', 'recovery'):
        if data.get('graph_file') in (None, '', 'YOUR_GRAPH_FILE'):
            errors.append('data.graph_file is not set (configs/base.yaml): the file name '
                          'you pass to retrieve_static_graph')
        if data.get('target', 'cases') != 'cases':
            errors.append("HHH4 tasks need data.target: cases (NB on raw counts)")

    q = data.get('quantiles') or []
    if q and (len(q) % 2 == 0 or abs(q[len(q) // 2] - 0.5) > 1e-9):
        errors.append('data.quantiles must be odd-length with 0.5 in the middle')

    bad_model = set(cfg.get('model', {})) - MODEL_KEYS
    if bad_model:
        errors.append(f'unknown model keys {sorted(bad_model)}; allowed: {sorted(MODEL_KEYS)}')
    for section, entries in [('evaluation.variants', (cfg.get('evaluation', {}) or {}).get('variants') or {}),
                             ('ablations.variants', (cfg.get('ablations', {}) or {}).get('variants') or {})]:
        for label, overrides in entries.items():
            bad = set(overrides or {}) - MODEL_KEYS
            if bad:
                errors.append(f'{section}.{label}: unknown model keys {sorted(bad)}')
    rc = cfg.get('hhh4_r', {}) or {}
    bad_r = set(rc) - {'enabled', 'rscript', 'nsim', 'seed', 'harmonics', 'max_lag', 'random_effects',
                       'power_law', 'family'}
    if bad_r:
        errors.append(f'unknown hhh4_r keys {sorted(bad_r)}')
    bad_py = set(cfg.get('hhh4_py', {}) or {}) - {'enabled', 'nsim', 'seed', 'harmonics', 'max_lag',
                                                  'random_effects', 'power_law'}
    if bad_py:
        errors.append(f'unknown hhh4_py keys {sorted(bad_py)}')
    bad_att = set(cfg.get('attribution', {}) or {}) - {'sources', 'replicates', 'side', 'years', 'lead',
                                                       'nsim', 'estimators', 'anchor_weights',
                                                       'template_coupling'}
    if bad_att:
        errors.append(f'unknown attribution keys {sorted(bad_att)}')
    bad_train = set(cfg.get('train', {})) - TRAIN_KEYS - RUNNER_TRAIN_KEYS
    if bad_train:
        errors.append(f'unknown train keys {sorted(bad_train)}; allowed: {sorted(TRAIN_KEYS | RUNNER_TRAIN_KEYS)}')

    ts = data.get('test_seasons')
    if ts is not None and (not isinstance(ts, list) or not all(isinstance(y, int) for y in ts)):
        errors.append('data.test_seasons must be a list of years, e.g. [2015, 2016, 2017, 2018] '
                      '(season = June of that year to June of the next)')

    seeds = cfg.get('train', {}).get('seeds', [0])
    if not isinstance(seeds, list) or not seeds:
        errors.append('train.seeds must be a non-empty list, e.g. [0, 1, 2]')

    if errors:
        raise ConfigError('Config problems:\n  - ' + '\n  - '.join(errors))


def run_name(cfg: dict) -> str:
    """Name for the output folder: ``name`` or task_disease_level_leadN."""
    if cfg.get('name'):
        return str(cfg['name'])
    d = cfg['data']
    if cfg['task'] == 'compare_diseases':
        return f"compare_diseases_{d['level']}_lead{d['lead']}"
    return f"{cfg['task']}_{d['disease']}_{d['level']}_lead{d['lead']}"


def epiconfig_kwargs(cfg: dict, disease: str | None = None) -> dict[str, Any]:
    """Arguments for ``EpiConfig`` from the ``data`` section."""
    d = cfg['data']
    target = d.get('target', 'cases')
    kw = dict(
        disease=disease or d['disease'], temporal_frequency='w', country=d.get('country', 'germany'),
        level=d['level'], horizon_size=1, horizon_leadtime=int(d['lead']),
        time_index_w=True, lag_column=d.get('lag_column', target), lag_num=int(d.get('lag_num', 1)),
        sequence_length=int(d.get('sequence_length', 4)),
        feature_popdens=bool(d.get('feature_popdens', False)),
        feature_popsize=bool(d.get('feature_popsize', False)),
        normalization_method=d.get('normalization_method', 'zscore'),
        log_transform=d.get('log_transform'),
        target_column=target, quantiles=list(d['quantiles']),
        **d['dates'],
    )
    kw.update(d.get('epiconfig', {}) or {})
    return kw


def dump(cfg: dict, path: str | Path) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with open(path, 'w') as f:
        yaml.safe_dump(clean, f, sort_keys=False, default_flow_style=None)
