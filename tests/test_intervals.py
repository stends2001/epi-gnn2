"""
Unit tests for the interval-forecasting additions. They run without the
epidemiological data: every test uses synthetic inputs or stubs.

    python -m pytest tests -q
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataloading.epiconfig.validator import EpiConfigValidator
from src.models.utils.conformal import residual_quantile_table
from src.models.utils.intervalmetrics import wis, coverage_and_width, quantile_ranks
from src.models.baselinemodels import SeasonalAverage
from src.models.gnnmodels.utils.lossmanager import LossManager
from src.models.gnnmodels.utils.lossmanager.nbloss import NBLoss
from src.models.gnnmodels.architectures.modules import (
    GCNModule, MonotoneQuantileHead, HHH4Module, nb_quantiles, sample_components,
)
from src.models.gnnmodels.gnnmodel.forecasting_mixin import GNNModelForecastMixin

QS = [0.05, 0.25, 0.5, 0.75, 0.95]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('qs, valid', [
    ([0.1, 0.5, 0.9], True),
    ([0.05, 0.25, 0.5, 0.75, 0.95], True),
    ([0.1, 0.5, 0.8], False),          # not symmetric
    ([0.9, 0.5, 0.1], False),          # not increasing
    ([0.1, 0.4, 0.5, 0.9], False),     # even length
])
def test_quantile_validation(qs, valid):
    cfg = SimpleNamespace(_prediction_mode='interval', quantiles=qs, _num_quantiles=len(qs))
    errors = EpiConfigValidator.__new__(EpiConfigValidator)
    errors.epiconfig = cfg
    found = errors._quantiles([])
    assert (found == []) == valid


# --------------------------------------------------------------------------- #
# conformal residual tables
# --------------------------------------------------------------------------- #
def test_residual_table_coverage_and_fallback():
    rng   = np.random.default_rng(0)
    n     = 20_000
    group = pd.Series(rng.integers(1, 53, n))
    pred  = pd.Series(rng.uniform(5, 50, n))
    # group 1 is made thin on purpose
    keep  = ~((group == 1) & (np.arange(n) % 50 != 0))
    group, pred = group[keep].reset_index(drop=True), pred[keep].reset_index(drop=True)
    target = pred + rng.normal(0, 3, len(pred))

    table = residual_quantile_table(target, pred, group, QS, scale='additive', min_obs=30)
    assert 1 in table.fallback_groups
    assert np.allclose(table.table.loc[1].to_numpy(), table.pooled.to_numpy())

    # fresh draws: coverage close to nominal
    pred2   = pd.Series(rng.uniform(5, 50, 20_000))
    group2  = pd.Series(rng.integers(1, 54, 20_000))     # 53 is unseen -> pooled
    target2 = pred2 + rng.normal(0, 3, 20_000)
    q = table.apply(pred2, group2, clip_lower=None)
    df = pd.DataFrame({'target': target2, **{f'pred_q{i+1}': q[l] for i, l in enumerate(QS)}})
    cw = coverage_and_width(df, QS)
    assert np.allclose(cw['coverage'], cw['nominal'], atol=0.03)


def test_log1p_scale_is_non_negative_and_relative():
    rng    = np.random.default_rng(1)
    size   = np.repeat([1.0, 100.0], 5000)              # small and large regions
    pred   = pd.Series(size * rng.uniform(1, 2, size.size))
    target = pd.Series(pred * np.exp(rng.normal(0, 0.2, size.size)))
    group  = pd.Series(np.zeros(size.size, dtype=int))

    table = residual_quantile_table(target, pred, group, QS, scale='log1p', min_obs=30)
    q = table.apply(pd.Series([0.0, 1.0, 100.0]), pd.Series([0, 0, 0]))
    assert (q.to_numpy() >= 0).all()
    widths = q[0.95] - q[0.05]
    assert widths[2] > 10 * widths[1]                    # scales with size
    assert widths[0] > 0                                  # no zero-width collapse at pred = 0


# --------------------------------------------------------------------------- #
# seasonal average helpers
# --------------------------------------------------------------------------- #
def _seasonal_stub():
    m = SeasonalAverage.__new__(SeasonalAverage)
    m.epiconfig = SimpleNamespace(id_column='node', temporal_column='timestamp')
    m._temporal_idx_column = 'tidx'
    m._year_column = '_year'
    return m


def test_seasonal_mean_fallback_and_leave_one_year_out():
    df = pd.DataFrame({
        'node':   [0, 0, 0, 0, 0],
        'tidx':   [10, 10, 10, 52, 53],
        '_year':  [2015, 2016, 2017, 2017, 2020],
        'target': [1.0, 2.0, 6.0, 4.0, 9.0],
        'train':  [True, True, True, True, False],
    })
    m = _seasonal_stub()

    pred = m._seasonal_mean_prediction(df)
    assert pred.notna().all()                            # left merge keeps every row
    assert pred.iloc[0] == pytest.approx(3.0)
    assert pred.iloc[4] == pytest.approx(4.0)            # week 53 -> week 52 of same node

    loo = m._leave_one_year_out_prediction(df)
    assert loo.iloc[0] == pytest.approx(4.0)             # mean of 2 and 6
    assert loo.iloc[2] == pytest.approx(1.5)             # mean of 1 and 2
    assert np.isnan(loo.iloc[3])                          # only one year in bin 52
    assert np.isnan(loo.iloc[4])                          # not a train row


# --------------------------------------------------------------------------- #
# pinball loss and quantile head
# --------------------------------------------------------------------------- #
def test_pinball_recovers_quantiles():
    torch.manual_seed(0)
    y = torch.randn(4000, 1)                             # N(0, 1)
    pred = torch.zeros(1, 1, len(QS), requires_grad=True)
    loss_fn = LossManager('pinball', quantiles=QS)
    opt = torch.optim.Adam([pred], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        loss = loss_fn(pred.expand(4000, 1, len(QS)), y)
        loss.backward()
        opt.step()
    expected = torch.tensor([-1.645, -0.674, 0.0, 0.674, 1.645])
    assert torch.allclose(pred.detach().view(-1), expected, atol=0.12)


def test_monotone_head_never_crosses():
    torch.manual_seed(0)
    head = MonotoneQuantileHead(8, horizon_size=3, quantiles=QS)
    out  = head(torch.randn(50, 8) * 10)
    assert out.shape == (50, 3, len(QS))
    assert (out[..., 1:] > out[..., :-1]).all()


def test_gcn_module_quantile_output_shape():
    N, F, S, H = 6, 3, 4, 2
    m = GCNModule(16, 2, 0.0, False, True, True, F, N, S, H, quantiles=QS)
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    out = m(torch.randn(N, F, S), ei)
    assert out.shape == (N, H, len(QS))


# --------------------------------------------------------------------------- #
# NB
# --------------------------------------------------------------------------- #
def test_nb_loss_matches_scipy():
    mu, alpha = torch.tensor([3.0, 40.0]), torch.tensor([0.2, 0.5])
    y = torch.tensor([1.0, 55.0])
    nll = NBLoss(reduction='none')((mu, alpha), y)
    r = 1 / alpha.numpy()
    ref = -nbinom.logpmf(y.numpy(), r, r / (r + mu.numpy()))
    assert np.allclose(nll.numpy(), ref, rtol=1e-5)


def test_nb_quantiles_and_component_sampling():
    torch.manual_seed(0)
    comps = {'endemic': torch.full((3, 2), 5.0),
             'epidemic': torch.full((3, 2), 10.0),
             'neighbourhood': torch.full((3, 2), 15.0)}
    alpha = torch.full((3, 2), 0.3)

    total, parts = sample_components(comps, alpha, n_samples=40_000)
    summed = sum(parts.values())
    assert torch.equal(summed, total)                    # parts add up exactly
    assert total.mean().item() == pytest.approx(30.0, rel=0.02)
    assert total.var().item() == pytest.approx(30 + 0.3 * 900, rel=0.05)
    assert parts['neighbourhood'].mean().item() == pytest.approx(15.0, rel=0.03)

    exact = nb_quantiles(np.full((3, 2), 30.0), np.full((3, 2), 0.3), QS)[0, 0]
    mc    = np.quantile(total[:, 0, 0].numpy(), QS)
    assert np.allclose(exact, mc, atol=1.5)


# --------------------------------------------------------------------------- #
# HHH4 module
# --------------------------------------------------------------------------- #
def _line_graph(n):
    src = torch.arange(n - 1)
    ei = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])
    return ei


def test_hhh4_components_sum_and_positive():
    N, F, S, H = 5, 3, 2, 2
    m = HHH4Module(N, S, H, incidence_idx=[0], endemic_idx=[1, 2], mu_init=60.0)
    (mu, alpha), parts = m(torch.randn(N, F, S), _line_graph(N), return_components=True)
    assert mu.shape == (N, H) and alpha.shape == (N, H)
    assert all((p >= 0).all() for p in parts.values())
    assert torch.allclose(sum(parts.values()), mu)
    assert mu.mean().item() == pytest.approx(60.0, rel=0.3)   # starts at the count scale


def test_hhh4_neighbourhood_ignores_own_incidence_even_with_self_loops():
    torch.manual_seed(0)
    N, F, S, H = 4, 2, 3, 1
    m = HHH4Module(N, S, H, incidence_idx=[0], endemic_idx=[1], num_layers=1).eval()

    # graph = line graph + explicit self-loops on every node
    ei = torch.cat([_line_graph(N), torch.arange(N).repeat(2, 1)], dim=1)
    x  = torch.randn(N, F, S)
    _, base = m(x, ei, return_components=True)

    x2 = x.clone()
    x2[0, 0, :] += 5.0                                    # change node 0's own incidence
    _, changed = m(x2, ei, return_components=True)

    assert torch.allclose(base['neighbourhood'][0], changed['neighbourhood'][0])   # own: unchanged
    assert not torch.allclose(base['neighbourhood'][1], changed['neighbourhood'][1])  # neighbour: changed
    assert not torch.allclose(base['epidemic'][0], changed['epidemic'][0])


def test_hhh4_identity_graph_gives_constant_neighbourhood():
    N, F, S, H = 4, 2, 3, 2
    m = HHH4Module(N, S, H, incidence_idx=[0], endemic_idx=[1]).eval()
    identity = torch.arange(N).repeat(2, 1)              # self-loops only -> no neighbours
    _, a = m(torch.randn(N, F, S), identity, return_components=True)
    _, b = m(torch.randn(N, F, S), identity, return_components=True)
    assert torch.allclose(a['neighbourhood'], b['neighbourhood'])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_wis_equals_summed_pinball_and_decomposes():
    rng = np.random.default_rng(3)
    y   = rng.normal(0, 1, 500)
    Q   = np.sort(rng.normal(0, 1, (500, len(QS))), axis=1)
    df  = pd.DataFrame({'target': y, **{f'pred_q{i+1}': Q[:, i] for i in range(len(QS))}})

    w = wis(df, QS, decompose=True)
    assert np.allclose(w['wis'], w[['dispersion', 'underprediction', 'overprediction']].sum(axis=1))

    e = y[:, None] - Q
    lv = np.asarray(QS)
    pinball = np.maximum(lv * e, (lv - 1) * e).sum(axis=1)
    K = len(QS) // 2
    assert np.allclose(w["wis"], pinball / (K + 0.5))

    r = quantile_ranks(df, QS)
    assert r.between(0, 1).all()


# --------------------------------------------------------------------------- #
# CQR on a stubbed model
# --------------------------------------------------------------------------- #
class _StubModel(GNNModelForecastMixin):
    def __init__(self, raw_val, y_val):
        self.epiconfig   = SimpleNamespace(quantiles=QS, _prediction_mode='interval')
        self.output_head = 'quantile'
        self.calibration_offsets = {}
        self._val = (raw_val, y_val)

    def _run_inference(self, dataset):
        return self._val[0], self._val[1], 0.0


def test_cqr_fixes_too_narrow_quantiles():
    rng = np.random.default_rng(4)
    T, N, H = 200, 20, 2
    z = np.array([-1.645, -0.674, 0.0, 0.674, 1.645])

    def make(seed):
        r = np.random.default_rng(seed)
        y = r.normal(0, 2.0, (T, N, H))                  # true sd = 2
        q = np.broadcast_to(z * 1.0, (T, N, H, len(QS))).copy()   # model thinks sd = 1
        return torch.tensor(q, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    q_val, y_val = make(10)
    q_test, y_test = make(11)
    m = _StubModel(q_val, y_val)
    adj = m._apply_cqr(q_test.numpy(), 'test')

    df = pd.DataFrame({'target': y_test.numpy().ravel(),
                       **{f'pred_q{i+1}': adj[..., i].ravel() for i in range(len(QS))}})
    cw = coverage_and_width(df, QS)
    assert np.allclose(cw['coverage'], cw['nominal'], atol=0.03)
    assert (np.diff(adj, axis=-1) >= 0).all()


# --------------------------------------------------------------------------- #
# training smoke test: HHH4Module + NB loss through the Strategy
# --------------------------------------------------------------------------- #
def test_hhh4_trains_with_nb_loss_through_strategy():
    from src.models.gnnmodels.utils import Strategy

    torch.manual_seed(0)
    N, S, H = 6, 2, 1
    ei = _line_graph(N)
    graph = SimpleNamespace(edge_index=ei, edge_weight=None)

    snapshots = []
    for _ in range(40):
        inc = torch.rand(N, 1, S) * 50
        season = torch.rand(N, 1, S)
        y = torch.poisson(inc[:, 0, -1:] * 0.8 + 5)      # depends on own last value
        snapshots.append(SimpleNamespace(x=torch.cat([inc, season], 1), y=y, graph=graph))

    m = HHH4Module(N, S, H, incidence_idx=[0], endemic_idx=[1], mu_init=25.0)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    loss_fn, strat = LossManager('nb'), Strategy()

    first = np.mean([strat.validation_step(m, s, loss_fn) for s in snapshots])
    for _ in range(30):
        for s in snapshots:
            strat.training_step(m, s, opt, loss_fn)
    last = np.mean([strat.validation_step(m, s, loss_fn) for s in snapshots])
    assert last < first
