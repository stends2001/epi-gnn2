"""
Shared setup for the run scripts: paths, the config, data builders and a
seeded HHH4 training helper.

Shared settings (graph file, dates, quantiles) are read from configs/base.yaml; every script imports
from here. Scripts can be run whole (``python scripts/02_hhh4.py``) or cell by cell
(``# %%`` markers work in VS Code and PyCharm as notebook cells).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# ---- make ``src`` importable from anywhere inside the repo ----
ROOT = Path(__file__).resolve().parent
while not (ROOT / 'src').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.dataloading import EpiConfig, EpiDataOrchestrator, BaseLineDataBuilder, GraphDataBuilder  # noqa: E402
from src.graphconstruction import identity_graph, rewired_graph  # noqa: E402

# =========================================================================== #
# Shared settings come from configs/base.yaml (the same file run.py uses), so
# the graph file and dates are set in one place.
# =========================================================================== #
from src.experiments.config import load_config  # noqa: E402

_BASE = load_config(ROOT / 'configs' / 'base.yaml', check=False)

GRAPH_FILE = _BASE['data']['graph_file']
QUANTILES  = list(_BASE['data']['quantiles'])
DATES      = dict(_BASE['data']['dates'])        # default: pre-COVID test season 2018/19
RESULTS    = ROOT / _BASE.get('output_dir', 'results')
# =========================================================================== #


def make_config(disease: str = 'norovirus',
                level: str = 'nuts3',
                lead: int = 4,
                sequence_length: int = 4,
                target: str = 'cases',
                dates: dict | None = None,
                **overrides) -> EpiConfig:
    """
    Config for count forecasts with intervals.

    With ``target='cases'`` the target and its lags stay raw counts (needed for
    the NB likelihood and the rate form of HHH4Model); only population features,
    if switched on, are normalised.
    """
    d = dict(DATES, **(dates or {}))
    cfg = dict(
        disease=disease, temporal_frequency='w', country='germany', level=level,
        horizon_size=1, horizon_leadtime=lead,
        time_index_w=True, lag_column=target, lag_num=1, sequence_length=sequence_length,
        feature_popdens=False, feature_popsize=False,
        normalization_method='zscore', log_transform=None,
        target_column=target, quantiles=QUANTILES, **d,
    )
    cfg.update(overrides)
    return EpiConfig(**cfg)


def build_data(cfg: EpiConfig):
    """Orchestrator and baseline data builder for a config."""
    edo = EpiDataOrchestrator(cfg).build()
    return edo, BaseLineDataBuilder(edo).build()


def graph_builder(edo, graph: str = 'real', seed: int = 0) -> GraphDataBuilder:
    """
    Graph data builder with the real graph or a control graph.

    graph : 'real', 'identity' or 'rewired' (degree-preserving; ``seed`` picks the draw).
    """
    gdb = GraphDataBuilder(edo)
    if GRAPH_FILE == 'YOUR_GRAPH_FILE':
        raise ValueError('Set data.graph_file in configs/base.yaml to your graph file name.')
    gdb.retrieve_static_graph(GRAPH_FILE)
    real = gdb.graph

    if graph == 'identity':
        gdb.use_graph(identity_graph(real.num_nodes))
    elif graph == 'rewired':
        gdb.use_graph(rewired_graph(real, seed=seed))
    elif graph != 'real':
        raise ValueError("graph must be 'real', 'identity' or 'rewired'")
    return gdb.build()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_hhh4(gdb: GraphDataBuilder, name: str, seed: int = 0,
               model_hparams: dict | None = None, global_hparams: dict | None = None):
    """Seeded HHH4Model: set hparams, train, forecast test (and val)."""
    from src.models.gnnmodels import HHH4Model

    set_seed(seed)
    m = HHH4Model(gdb, name=name)
    m.set_model_hparams(**{**dict(alpha_mode='node'), **(model_hparams or {})})
    m.set_global_hparams(**{**dict(lr=5e-3, n_epochs=300, patience=30), **(global_hparams or {})})
    m.train()
    m.forecast('test')
    return m


def fit_baselines(db: BaseLineDataBuilder) -> dict:
    """Persistence and SeasonalAverage (log1p residuals) forecasted on test."""
    from src.models import Persistence, SeasonalAverage

    out = {
        'persistence':      Persistence(db, name='persistence', residual_scale='log1p'),
        'seasonal_average': SeasonalAverage(db, name='seasonal_average', residual_scale='log1p'),
    }
    for m in out.values():
        m.forecast('test')
    return out


def results_dir(*parts: str) -> Path:
    p = RESULTS.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
