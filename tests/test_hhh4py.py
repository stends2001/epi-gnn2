"""hhh4 in Python: basic behaviour, and agreement with surveillance::hhh4 when R is installed."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.statistical import HHH4Py, neighbourhood_order
from src.experiments.simulators import grid_adjacency, simulate_sir_network


@pytest.fixture(scope='module')
def data():
    A = grid_adjacency(3)
    counts, truth, pop = simulate_sir_network(A, years=4, coupling=0.3, seed=0)
    return A, counts, truth, pop


def test_neighbourhood_order():
    o = neighbourhood_order(grid_adjacency(3), max_lag=5)
    assert o[0, 1] == 1 and o[0, 4] == 2 and o[0, 8] == 4 and o[0, 0] == 0


def test_fit_simulate_scenarios(data):
    A, counts, _, pop = data
    h = HHH4Py(harmonics=1, random_effects=True).fit(counts.to_numpy(), A, pop, fit_rows=np.arange(1, 150))
    s = h.summary()
    assert np.isfinite(s['loglik']) and s['overdispersion_psi'] > 0
    c = h.fitted_components(np.arange(150, 200))
    assert np.allclose(c[['endemic', 'epidemic', 'neighbourhood']].sum(axis=1), c['mean'])
    draws = h.simulate([150, 160], lead=3, nsim=30, seed=0)
    assert draws.shape == (2, 30, 9) and (draws >= 0).all()
    y, comp = h.simulate_series('no_ne', seed=0)
    if y is not None:
        assert comp['neighbourhood'].max() < 1e-8


@pytest.mark.skipif(shutil.which('Rscript') is None, reason='R not installed')
def test_matches_surveillance_without_random_effects(tmp_path, data):
    from src.experiments.hhh4r import run_hhh4_r
    A, counts, _, pop = data
    fit_end_row = 150
    res = run_hhh4_r(counts, A, pop, counts.index[fit_end_row - 1], [counts.index[160]], 1,
                     [0.1, 0.5, 0.9], tmp_path, spec={'random_effects': False, 'nsim': 10})
    info = dict(zip(res['fit_info']['key'], res['fit_info']['value']))
    h = HHH4Py(harmonics=1, random_effects=False).fit(counts.to_numpy(), A, pop,
                                                     fit_rows=np.arange(1, fit_end_row))
    assert h.summary()['loglik'] == pytest.approx(float(info['loglik']), abs=0.5)
    rc = res['components']
    pc = h.fitted_components(rc['row'].unique() - 1)
    assert np.corrcoef(np.sort(rc['mean']), np.sort(pc['mean']))[0, 1] > 0.999
