"""
Simulators with a known endemic / epidemic / neighbourhood split, for testing
whether a model attributes cases to the right route.

``simulate_sir_network`` is model-neutral: a seasonal SIR process per region,
coupled through the graph, observed as reported counts. It is neither hhh4 nor
the neural model, so neither has a home advantage. Its true one-step components
(in expected reported cases) are

- endemic:       background cases, constant per region
- epidemic:      new cases from infections in the region itself
- neighbourhood: new cases from infections in neighbouring regions

``HHH4Py.simulate_series`` gives the hhh4-generated counterpart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def grid_adjacency(side: int) -> np.ndarray:
    n = side * side
    A = np.zeros((n, n), dtype=int)
    for i in range(side):
        for j in range(side):
            k = i * side + j
            if j < side - 1:
                A[k, k + 1] = A[k + 1, k] = 1
            if i < side - 1:
                A[k, k + side] = A[k + side, k] = 1
    return A


def simulate_sir_network(adjacency: np.ndarray,
                         years: int = 9,
                         coupling: float = 0.3,
                         seed: int = 0,
                         start: str = '2010-07-05',
                         reporting: float = 0.01,
                         background: float = 0.5,
                         beta_range: tuple[float, float] = (1.5, 2.3),
                         start_weeks: tuple[int, int] = (18, 30),
                         susceptible_range: tuple[float, float] = (0.5, 0.8),
                         pop_range: tuple[float, float] = (5e4, 4e5),
                         seed_prob: float = 0.4,
                         seed_size: float = 30.0):
    """
    Yearly epidemic waves (random start week, transmission rate and susceptible
    fraction per season; each region independently seeded with probability
    ``seed_prob`` at a random delay) spreading within regions and, with weight
    ``coupling``, to graph neighbours. With coupling 0 regions share the season
    but never infect each other: the hard case for attributing spread.

    Returns
    -------
    counts : DataFrame [weeks x regions] of reported cases (index: Monday dates)
    truth  : DataFrame date, node, endemic, epidemic, neighbourhood, mean - the
             expected reported cases of each week split by route
    population : array [regions]
    """
    rng = np.random.default_rng(seed)
    A = (np.asarray(adjacency) > 0).astype(float)
    np.fill_diagonal(A, 0)
    Anorm = A / np.maximum(A.sum(1, keepdims=True), 1)
    N = A.shape[0]
    weeks = years * 52
    dates = pd.date_range(start, periods=weeks, freq='7D')
    pop = rng.uniform(*pop_range, N)

    y = rng.poisson(background, (weeks, N)).astype(float)
    comp = np.zeros((3, weeks, N))
    comp[0] = background

    for yr in range(years):
        t_start = yr * 52 + rng.integers(*start_weeks)
        beta = rng.uniform(*beta_range)
        S = pop * rng.uniform(*susceptible_range, N)
        I = np.zeros(N)
        # independent introductions: each region is seeded with probability
        # seed_prob, at a random delay; with coupling = 0 the epidemic then stays
        # in the seeded regions, with coupling > 0 it also spreads to the others
        seeded = rng.random(N) < seed_prob
        seeded[rng.integers(N)] = True
        delay = np.where(seeded, rng.integers(0, 8, N), -1)
        for t in range(t_start, min(t_start + 35, weeks - 1)):
            intro = (delay == t - t_start)
            I = I + intro * seed_size
            prev = I / pop
            own = beta * S * (1 - coupling) * prev
            ne = beta * S * coupling * (Anorm @ prev)
            force = own + ne
            new = np.minimum(S, rng.poisson(force)).astype(float)
            S -= new
            I = new
            comp[1, t + 1] += reporting * own
            comp[2, t + 1] += reporting * ne
            y[t + 1] += rng.poisson(reporting * new)

    counts = pd.DataFrame(y, index=dates)
    truth = pd.DataFrame({'date': np.repeat(dates, N), 'node': np.tile(np.arange(N), weeks),
                          'endemic': comp[0].ravel(), 'epidemic': comp[1].ravel(),
                          'neighbourhood': comp[2].ravel()})
    truth['mean'] = truth[['endemic', 'epidemic', 'neighbourhood']].sum(axis=1)
    return counts, truth, pop
