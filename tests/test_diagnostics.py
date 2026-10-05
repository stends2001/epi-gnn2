"""
Integration tests for the diagnostics, the node-specific HHH4 parameters and the
graph controls. A synthetic "HHH4Model" is assembled from real components
(HHH4Module, PredictionCollection, GraphStructure) without the data pipeline.

    python -m pytest tests -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graphconstruction import GraphStructure, identity_graph, rewired_graph, degree_sequence
from src.dataloading.databuilders.graphdatabuilder.datacontainers import Data
from src.models.utils.predictioncollection import PredictionCollection
from src.models.gnnmodels.architectures.modules import HHH4Module, nb_quantiles
from src.models.gnnmodels.architectures.modelarchitectures.hhh4model import HHH4Model
from src.models.gnnmodels.utils import LossManager, Strategy
from src.models.utils.intervalmetrics import (
    evaluate_model_intervals, compare_models, season_mask,
)
from src.models import diagnostics as dg

QS = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
N, S, H, LEAD = 9, 2, 1, 1
WEEKS = 160


# --------------------------------------------------------------------------- #
# synthetic world: 3x3 grid of regions, seasonal counts with spatial spread
# --------------------------------------------------------------------------- #
def _grid_graph(side=3):
    e = []
    for i in range(side):
        for j in range(side):
            k = i * side + j
            if j < side - 1: e += [(k, k + 1), (k + 1, k)]
            if i < side - 1: e += [(k, k + side), (k + side, k)]
    return GraphStructure.from_list(e, [1.0] * len(e), side * side)


def _simulate(seed=0, coupling=0.0):
    """Endemic seasonal level + own-region autoregression (+ spread from neighbours)."""
    rng = np.random.default_rng(seed)
    A = (_grid_graph().adjacency_matrix.numpy() > 0).astype(float)
    t0 = pd.date_range('2015-01-05', periods=WEEKS, freq='7D')
    week = t0.isocalendar().week.to_numpy().astype(float)
    size = rng.uniform(20, 120, N)
    season = 1 + 0.8 * np.cos(2 * np.pi * (week - 2) / 52)
    y = np.zeros((WEEKS, N))
    y[0] = size * season[0]
    for t in range(1, WEEKS):
        lam = 0.3 * size * season[t] + (0.6 - coupling) * y[t - 1] + coupling * (A @ y[t - 1]) / A.sum(1)
        y[t] = rng.poisson(lam)
    return t0, week, y


class _Coll:
    def __init__(self):
        self.c = {'train': PredictionCollection(), 'val': PredictionCollection(), 'test': PredictionCollection()}

    def get_preds(self, d):
        return self.c[d]


def _snapshots(week, y, graph):
    """x: [N, 3 features (cases, sin, cos), S]; y: cases at t0 + LEAD."""
    snaps, t_index = [], []
    sinw, cosw = np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52)
    for t in range(S - 1, WEEKS - LEAD):
        x = np.stack([y[t - S + 1:t + 1].T,
                      np.tile(sinw[t - S + 1:t + 1], (N, 1)),
                      np.tile(cosw[t - S + 1:t + 1], (N, 1))], axis=1)
        snaps.append(Data(torch.tensor(x, dtype=torch.float32),
                          torch.tensor(y[t + LEAD], dtype=torch.float32).view(N, 1), graph))
        t_index.append(t)
    return snaps, np.array(t_index)


@pytest.fixture(scope='module')
def fitted():
    return _build(coupling=0.0)


def _build(coupling=0.0, epochs=60):
    torch.manual_seed(0)
    t0, week, y = _simulate(coupling=coupling)
    graph = _grid_graph()
    snaps, t_idx = _snapshots(week, y, graph)
    n_train = int(0.6 * len(snaps)); n_val = int(0.2 * len(snaps))
    splits = {'train': snaps[:n_train], 'val': snaps[n_train:n_train + n_val],
              'test': snaps[n_train + n_val:]}
    t_splits = {'train': t_idx[:n_train], 'val': t_idx[n_train:n_train + n_val],
                'test': t_idx[n_train + n_val:]}

    m = HHH4Model.__new__(HHH4Model)
    m.name = 'hhh4_synthetic'
    m.epiconfig = SimpleNamespace(quantiles=QS, horizon_size=H, horizon_leadtime=LEAD,
                                  temporal_column='timestamp', id_column='node', disease='norovirus',
                                  level='nuts3', temporal_frequency='w', pred_column='pred',
                                  target_column='cases', sequence_length=S, _prediction_mode='interval',
                                  _num_quantiles=len(QS))
    m.device = torch.device('cpu')
    m.status_dict = {'model_hparams_set': True, 'global_hparams_set': True, 'trained': True}
    m.incidence_features = ['cases_lag0']
    m.endemic_features = ['tt_sin_w', 'tt_cos_w']
    m._get_dataloader = lambda d: splits[d]
    m._t0_timestamps = lambda d, T: t0[t_splits[d][:T]].values

    import geopandas as gpd
    from shapely.geometry import box
    m.context_data = SimpleNamespace(
        nodenames=pd.DataFrame({'node': range(N), 'nuts3_name': [f'R{i}' for i in range(N)]}),
        local_shapedata=gpd.GeoDataFrame({'node': range(N)},
                                         geometry=[box(i % 3, i // 3, i % 3 + 1, i // 3 + 1) for i in range(N)]),
        num_nodes=N)

    train_y = torch.stack([s.y for s in splits['train']])
    node_means = train_y.mean(dim=(0, 2)).numpy()
    m.model = HHH4Module(N, S, H, incidence_idx=[0], endemic_idx=[1, 2], hidden_size=8,
                         mu_init=float(node_means.mean()), node_means=node_means)

    opt, loss, strat = torch.optim.Adam(m.model.parameters(), lr=0.005), LossManager('nb'), Strategy()
    for _ in range(epochs):
        for s in splits['train']:
            strat.training_step(m.model, s, opt, loss)

    # predictions in the PredictionManager layout (timestamps at target time)
    m.predictions = _Coll()
    for d in ('train', 'val', 'test'):
        comp = m.forecast_components(d)
        qv = nb_quantiles(comp['mu'].to_numpy(), comp['alpha'].to_numpy(), QS)
        df = pd.DataFrame({'timestamp': pd.to_datetime(comp['timestamp']) + pd.Timedelta(weeks=LEAD),
                           'node': comp['node'], 'target': comp['target']})
        for i in range(len(QS)):
            df[f'pred_q{i+1}'] = qv[:, i]
        m.predictions.get_preds(d).add(df, 0, True, False)

    # persistence-like baseline on the same frames
    base = SimpleNamespace(name='persistence', epiconfig=m.epiconfig, predictions=_Coll())
    for d in ('train', 'val', 'test'):
        df = m.predictions.get_preds(d).get(0, True, False).sort_values(['node', 'timestamp']).copy()
        last = df.groupby('node')['target'].shift(LEAD).fillna(df['target'])
        for i, q in enumerate(QS):
            df[f'pred_q{i+1}'] = np.clip(last * (1 + 0.6 * (q - 0.5)), 0, None)
        base.predictions.get_preds(d).add(df.reset_index(drop=True), 0, True, False)
    return m, base, y


# --------------------------------------------------------------------------- #
def test_node_parameters_and_seasonality(fitted):
    m, _, _ = fitted
    p = m.node_parameters()
    assert len(p) == N
    for c in ['endemic_baseline', 'seasonal_amplitude', 'endemic_peak_week',
              'epidemic_multiplier', 'neighbourhood_multiplier', 'alpha', 'node_name']:
        assert c in p.columns
    assert p['endemic_peak_week'].between(1, 53).all()
    assert (p['seasonal_amplitude'] >= 1).all()


def test_regularization_is_positive_and_scaled():
    mod = HHH4Module(N, S, H, incidence_idx=[0], endemic_idx=[1, 2], node_penalty=0.5)
    assert mod.regularization().item() == 0.0              # all deviations start at 0
    with torch.no_grad():
        mod.end_node_coef.fill_(1.0)
    assert mod.regularization().item() == pytest.approx(0.0)   # a common shift is not a node effect
    with torch.no_grad():
        mod.end_node_coef.zero_()
        mod.end_node_coef[0] = 1.0
    assert mod.regularization().item() > 0


def test_scale_keeps_large_counts_trainable():
    mod = HHH4Module(4, 2, 1, incidence_idx=[0], endemic_idx=[1], mu_init=5000.0,
                     node_means=[1000, 3000, 6000, 10000])
    x = torch.zeros(4, 2, 2); x[:, 0, :] = 5000
    (mu, _), parts = mod(x, torch.zeros(2, 0, dtype=torch.long), return_components=True)
    assert torch.allclose(parts['endemic'][:, 0], torch.tensor([1000, 3000, 6000, 10000.]) / 3, rtol=1e-3)
    assert mu.min() > 100


def test_node_effects_are_relative(fitted):
    m, _, _ = fitted
    p = m.node_parameters()
    for c in ('epidemic_multiplier', 'neighbourhood_multiplier'):
        assert np.exp(np.log(p[c]).mean()) == pytest.approx(1.0, rel=1e-4)


def test_neighbourhood_branch_tracks_simulated_spread(fitted):
    """
    With spread between regions the neighbourhood share is much larger than
    without. Without spread it is NOT zero: neighbours share the seasonal curve,
    so their mean also predicts the seasonal level. The absolute share therefore
    does not measure transmission; the contrast (or a graph control) does.
    """
    m0, _, _ = fitted
    m1, _, _ = _build(coupling=0.4)
    s0 = dg.component_table(m0, 'test').query("period == 'all'")['share_neighbourhood'].iloc[0]
    s1 = dg.component_table(m1, 'test').query("period == 'all'")['share_neighbourhood'].iloc[0]
    assert s0 < 0.3
    assert s1 > s0 + 0.15


def test_component_tables(fitted):
    m, _, _ = fitted
    ct = dg.component_table(m, 'test')
    assert set(ct['period']) <= {'all', 'in-season', 'off-season'}
    row = ct[ct['period'] == 'all'].iloc[0]
    assert row[['share_endemic', 'share_epidemic', 'share_neighbourhood']].sum() == pytest.approx(1, abs=1e-6)
    byn = dg.component_by_node(m, 'test')
    assert len(byn) == N and 'endemic_peak_week' in byn.columns
    cmp_ = dg.compare_components({'a': m, 'b': m}, 'test', period='all')
    assert list(cmp_['model']) == ['a', 'b']


def test_season_filter_and_comparison(fitted):
    m, base, _ = fitted
    full = evaluate_model_intervals(m, 'test')
    ins  = evaluate_model_intervals(m, 'test', season='in')
    assert ins['n'].iloc[0] < full['n'].iloc[0]
    tab = compare_models({'hhh4': m, 'persistence': base}, 'test', reference='persistence')
    assert tab.loc[tab['model'] == 'persistence', 'rel_wis'].iloc[0] == pytest.approx(1.0)
    assert {'cov95', 'width50'} <= set(tab.columns)
    wk = season_mask(pd.Series(pd.to_datetime(['2020-01-06', '2020-07-06'])), 40, 15)
    assert list(wk) == [True, False]


def test_sanity_report_flags(fitted):
    m, base, _ = fitted
    rep = dg.sanity_report(m, baselines={'persistence': base}, dataset='test')
    assert {'check', 'value', 'status'} <= set(rep.columns)
    status = dict(zip(rep['check'], rep['status']))
    assert status['missing values in predictions'] == 'PASS'
    assert status['rows with crossing quantiles'] == 'PASS'
    assert status['components add up to mu'] == 'PASS'
    dg.print_sanity(rep)

    # the persistence baseline copies the last value: lag check must catch it
    lc_base = dg.lag_correlation(base, 'test')
    assert int(lc_base.loc[lc_base['corr'].idxmax(), 'lag_weeks']) == LEAD


def test_all_plots_render(fitted):
    m, base, _ = fitted
    figs = [
        dg.plot_decomposition(m, nodes=[0, 4]),
        dg.plot_component_shares(m),
        dg.plot_node_maps(m),
        dg.plot_seasonal_curves(m),
        dg.plot_calibration({'hhh4': m, 'persistence': base}),
        dg.plot_lag_check({'hhh4': m, 'persistence': base}),
        dg.plot_pred_vs_obs(m),
        dg.plot_model_comparison(compare_models({'hhh4': m, 'persistence': base}, 'test')),
    ]
    for f in figs:
        assert f is not None and len(f.axes) > 0


# --------------------------------------------------------------------------- #
def test_graph_controls():
    g = _grid_graph(4)
    r = rewired_graph(g, seed=3)
    assert np.array_equal(degree_sequence(g), degree_sequence(r))
    real = set(map(tuple, g.edge_index.t().tolist()))
    new  = set(map(tuple, r.edge_index.t().tolist()))
    assert len(real & new) / len(real) < 0.6                  # geography largely scrambled
    assert all((b, a) in new for a, b in new)                  # still undirected
    ident = identity_graph(5)
    assert ident.num_edges == 5 and (ident.edge_index[0] == ident.edge_index[1]).all()


# --------------------------------------------------------------------------- #
# case-count configuration
# --------------------------------------------------------------------------- #
def _input_errors(**kw):
    from src.dataloading.epiconfig.validator import EpiConfigValidator
    base = dict(horizon_size=1, horizon_leadtime=1, sequence_length=1, lag_num=1,
                country='germany', level='nuts3', target_column='cases',
                lag_column='cases', log_transform=None)
    v = EpiConfigValidator.__new__(EpiConfigValidator)
    v.epiconfig = SimpleNamespace(**{**base, **kw})
    return v._input([])


def test_case_target_config_rules():
    assert _input_errors() == []
    assert _input_errors(target_column='incidence', lag_column='incidence') == []
    assert _input_errors(target_column='cases', lag_column='incidence') == []
    assert len(_input_errors(target_column='incidence', lag_column='cases')) == 1   # cases dropped
    assert len(_input_errors(log_transform=['cases'])) == 1                          # counts stay raw
    assert len(_input_errors(target_column='deaths')) == 1


def test_case_target_is_registered_untransformed():
    from src.dataloading.epidataorchestration.orchestrator import EpiDataOrchestrator
    for target, transformed in [('cases', False), ('incidence', True)]:
        cfg = SimpleNamespace(temporal_column='timestamp', id_column='node', target_column=target)
        edo = EpiDataOrchestrator(cfg)
        assert edo.column_registration.get_entry_by_name('target').transformation is transformed
