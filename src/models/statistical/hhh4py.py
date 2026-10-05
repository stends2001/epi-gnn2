"""
hhh4 (Held, Hoehle & Hofmann 2005; Meyer, Held & Hoehle 2017) in PyTorch.

A Python re-implementation of the endemic-epidemic model of the R package
``surveillance``, so the reference model runs where R is not available. It was
checked against ``surveillance::hhh4`` on the same data (log-likelihood, fitted
means and coefficients; see tests/test_hhh4py.py).

Model, for region i and week t::

    Y_it | past ~ NegBin(mu_it, psi),   Var = mu + psi mu^2
    mu_it  = e_i nu_it  +  lambda_it Y_i,t-1  +  phi_it sum_j w_ji Y_j,t-1
    log nu_it     = a_e + b_e,i + sum_s (c_es sin(2 pi s t / 52) + d_es cos(...))
    log lambda_it = a_a + b_a,i + seasonal terms
    log phi_it    = a_n + b_n,i + seasonal terms

- ``e_i``: population fraction (endemic offset).
- ``b``: region random intercepts, b_k ~ N(0, sigma_k^2) per component, fitted by
  penalised likelihood with sigma_k^2 updated by a Laplace (EM-type) step,
  ``sigma_k^2 = (|b_k|^2 + tr(H^-1)_kk) / N``. surveillance additionally allows a
  correlation between the three components (``corr = "all"``); this version uses
  independent components.
- ``w_ji``: power-law weights o_ji^(-d) in the neighbourhood order o_ji (1 =
  direct neighbours, up to ``max_lag``), normalised so each source region's
  weights sum to 1 (``W_powerlaw(normalize = TRUE)``).
- Without random effects: region-specific fixed endemic intercepts and no region
  effects in the epidemic / neighbourhood parts (the usual fixed-effects hhh4).

Fitting is full-batch L-BFGS on the exact likelihood. Multi-week forecasts are
made by simulating the fitted process forward from each forecast origin, as
``simulate.hhh4`` does; components are one-step-ahead means given the past.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #
def neighbourhood_order(adjacency: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """Shortest-path order between regions (0 on the diagonal and beyond max_lag)."""
    A = (np.asarray(adjacency) > 0).astype(int)
    A = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(A, 0)
    n = A.shape[0]
    order = np.zeros((n, n), dtype=int)
    reach = np.eye(n, dtype=bool)
    frontier = np.eye(n, dtype=bool)
    for k in range(1, max_lag + 1):
        frontier = (frontier.astype(int) @ A > 0) & ~reach
        order[frontier] = k
        reach |= frontier
        if not frontier.any():
            break
    return order


def _seasonal_design(t: np.ndarray, harmonics: int, period: int = 52) -> np.ndarray:
    cols = []
    for s in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * s * t / period), np.cos(2 * np.pi * s * t / period)]
    return np.stack(cols, axis=1) if cols else np.zeros((len(t), 0))


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@dataclass
class HHH4Fit:
    params: dict
    sigma2: dict
    loglik: float
    penalised_loglik: float
    n_iter: int
    converged: bool


class HHH4Py:
    """
    hhh4 in PyTorch.

    Parameters
    ----------
    harmonics : int
        Sin/cos pairs per component (``S`` in ``addSeason2formula``).
    random_effects : bool
        Region random intercepts in all three components.
    power_law : bool
        Power-law neighbourhood weights (else first-order neighbours, weight 1).
    max_lag : int
        Highest neighbourhood order with a non-zero power-law weight.
    period : int
        Seasonal period in weeks.
    """
    components = ('endemic', 'epidemic', 'neighbourhood')

    def __init__(self, harmonics: int = 1, random_effects: bool = True, power_law: bool = True,
                 max_lag: int = 5, period: int = 52, dtype=torch.float64):
        self.harmonics = harmonics
        self.random_effects = random_effects
        self.power_law = power_law
        self.max_lag = max_lag
        self.period = period
        self.dtype = dtype
        self.fit_: HHH4Fit | None = None

    # ------------------------------------------------------------------ #
    def _setup(self, Y: np.ndarray, adjacency: np.ndarray, population: np.ndarray):
        T, N = Y.shape
        self.N, self.T = N, T
        self.Y = torch.tensor(Y, dtype=self.dtype)
        pop = np.asarray(population, dtype=float)
        self.log_e = torch.tensor(np.log(pop / pop.sum()), dtype=self.dtype)
        order = neighbourhood_order(adjacency, self.max_lag)
        self.order = torch.tensor(order, dtype=self.dtype)
        self.mask = self.order > 0
        # seasonality on the row index, as hhh4 uses t
        self.Xs = torch.tensor(_seasonal_design(np.arange(T), self.harmonics, self.period), dtype=self.dtype)

    def _init_params(self) -> dict:
        S2 = self.Xs.shape[1]
        y = self.Y
        mean_i = y.mean(0).clamp(min=0.1)
        p = {
            'end_int': torch.tensor([float(torch.log(mean_i.mean() * 0.3 / torch.exp(self.log_e).mean()))]),
            'end_seas': torch.zeros(S2),
            'ar_int': torch.tensor([np.log(0.3)]),
            'ar_seas': torch.zeros(S2),
            'ne_int': torch.tensor([np.log(0.2)]),
            'ne_seas': torch.zeros(S2),
            'log_d': torch.tensor([0.0]),
            'log_psi': torch.tensor([np.log(0.3)]),
            'b_end': torch.log(mean_i * 0.3) - self.log_e - torch.log(mean_i.mean() * 0.3 / torch.exp(self.log_e).mean()),
            'b_ar': torch.zeros(self.N),
            'b_ne': torch.zeros(self.N),
        }
        p['b_end'] = p['b_end'] - p['b_end'].mean()
        if not self.random_effects:
            p['b_end'] = p['b_end'] + p['end_int']           # fixed region intercepts
            p['end_int'] = torch.zeros(1)
        return {k: v.to(self.dtype) for k, v in p.items()}

    def _free(self) -> list[str]:
        names = ['end_int', 'end_seas', 'ar_int', 'ar_seas', 'ne_int', 'ne_seas', 'log_psi', 'b_end']
        if self.power_law:
            names.append('log_d')
        if self.random_effects:
            names += ['b_ar', 'b_ne']
        else:
            names.remove('end_int')
        return names

    def weights(self, p: dict) -> torch.Tensor:
        """W[j, i]: weight of source region j for target region i (rows sum to 1)."""
        if not self.power_law:
            return self.mask.to(self.dtype)
        d = torch.exp(p['log_d'])
        w = torch.where(self.mask, self.order.clamp(min=1.0) ** (-d), torch.zeros_like(self.order))
        return w / w.sum(1, keepdim=True).clamp(min=1e-12)

    def rates(self, p: dict, t_idx: torch.Tensor):
        """log nu, log lambda, log phi at rows t_idx: each [len(t_idx), N]."""
        Xs = self.Xs[t_idx]
        nu  = p['end_int'] + Xs @ p['end_seas'] if Xs.shape[1] else p['end_int'].expand(len(t_idx))
        lam = p['ar_int'] + Xs @ p['ar_seas'] if Xs.shape[1] else p['ar_int'].expand(len(t_idx))
        phi = p['ne_int'] + Xs @ p['ne_seas'] if Xs.shape[1] else p['ne_int'].expand(len(t_idx))
        log_nu  = nu.view(-1, 1) + p['b_end'].view(1, -1) + self.log_e.view(1, -1)
        log_lam = lam.view(-1, 1) + p['b_ar'].view(1, -1)
        log_phi = phi.view(-1, 1) + p['b_ne'].view(1, -1)
        return log_nu, log_lam, log_phi

    def mean_components(self, p: dict, t_idx: torch.Tensor, y_prev: torch.Tensor):
        """One-step components at rows t_idx given the previous week's counts y_prev [len, N]."""
        log_nu, log_lam, log_phi = self.rates(p, t_idx)
        W = self.weights(p)
        endemic = torch.exp(log_nu)
        epidemic = torch.exp(log_lam) * y_prev
        neighbourhood = torch.exp(log_phi) * (y_prev @ W)
        return endemic, epidemic, neighbourhood

    @staticmethod
    def _nb_loglik(y, mu, psi):
        r = 1.0 / psi
        mu = mu.clamp(min=1e-10)
        return (torch.lgamma(y + r) - torch.lgamma(y + 1) - torch.lgamma(r)
                + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu)))

    def loglik(self, p: dict, rows: torch.Tensor) -> torch.Tensor:
        e, a, n = self.mean_components(p, rows, self.Y[rows - 1])
        return self._nb_loglik(self.Y[rows], e + a + n, torch.exp(p['log_psi'])).sum()

    # ------------------------------------------------------------------ #
    def fit(self, Y, adjacency, population, fit_rows=None, max_outer: int = 8,
            lbfgs_iter: int = 300, tol: float = 1e-6, verbose: bool = False) -> 'HHH4Py':
        """
        Fit on target rows ``fit_rows`` (0-based row indices >= 1; default: all rows
        from 1). Rows after the fit window are never used.
        """
        Y = np.asarray(Y, dtype=float)
        self._setup(Y, adjacency, population)
        rows = torch.arange(1, self.T) if fit_rows is None else torch.as_tensor(np.asarray(fit_rows), dtype=torch.long)
        if rows.min() < 1:
            raise ValueError('fit rows must be >= 1 (row 0 has no previous week)')

        p = self._init_params()
        free = self._free()
        sizes = [p[k].numel() for k in free]
        sigma2 = {'b_end': 1.0, 'b_ar': 1.0, 'b_ne': 1.0}

        def unflat(v):
            out, i = dict(p), 0
            for k, s in zip(free, sizes):
                out[k] = v[i:i + s].view(p[k].shape)
                i += s
            return out

        def objective(v):
            q = unflat(v)
            nll = -self.loglik(q, rows)
            if self.random_effects:
                for k in ('b_end', 'b_ar', 'b_ne'):
                    nll = nll + 0.5 * (q[k] ** 2).sum() / sigma2[k]
            return nll

        v = torch.cat([p[k].reshape(-1) for k in free]).clone().requires_grad_(True)
        converged, prev = False, None
        for outer in range(max_outer if self.random_effects else 1):
            opt = torch.optim.LBFGS([v], lr=1.0, max_iter=lbfgs_iter, tolerance_grad=1e-7,
                                    tolerance_change=1e-10, history_size=50, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                f = objective(v)
                f.backward()
                return f

            opt.step(closure)
            if not self.random_effects:
                converged = True
                break
            # Laplace / EM update of the random-effect variances
            H = torch.autograd.functional.hessian(objective, v.detach())
            Hinv = torch.linalg.pinv(H)
            q = unflat(v.detach())
            new = {}
            offset = 0
            for k, s in zip(free, sizes):
                if k in sigma2:
                    block = Hinv[offset:offset + s, offset:offset + s]
                    new[k] = float(((q[k] ** 2).sum() + torch.trace(block)) / s)
                offset += s
            change = max(abs(np.log(max(new[k], 1e-8)) - np.log(max(sigma2[k], 1e-8))) for k in sigma2)
            sigma2 = {k: max(val, 1e-6) for k, val in new.items()}
            if verbose:
                print(f'  outer {outer}: sigma2 {sigma2}, change {change:.4f}')
            if change < 0.01:
                converged = True
                break

        q = {k: t.detach() for k, t in unflat(v.detach()).items()}
        ll = float(self.loglik(q, rows))
        pen = ll - (0.5 * sum(float((q[k] ** 2).sum()) / sigma2[k] for k in sigma2) if self.random_effects else 0.0)
        self.fit_ = HHH4Fit(params=q, sigma2=sigma2 if self.random_effects else {}, loglik=ll,
                            penalised_loglik=pen, n_iter=outer + 1, converged=converged)
        self.fit_rows = rows
        return self

    # ------------------------------------------------------------------ #
    @property
    def params(self) -> dict:
        if self.fit_ is None:
            raise RuntimeError('call fit() first')
        return self.fit_.params

    def fitted_components(self, rows) -> pd.DataFrame:
        """One-step components at target rows given the OBSERVED previous week."""
        rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        with torch.no_grad():
            e, a, n = self.mean_components(self.params, rows, self.Y[rows - 1])
        R, N = e.shape
        return pd.DataFrame({'row': np.repeat(rows.numpy(), N), 'node': np.tile(np.arange(N), R),
                             'endemic': e.reshape(-1).numpy(), 'epidemic': a.reshape(-1).numpy(),
                             'neighbourhood': n.reshape(-1).numpy(),
                             'mean': (e + a + n).reshape(-1).numpy(), 'target': self.Y[rows].reshape(-1).numpy()})

    def simulate(self, origins, lead: int, nsim: int = 500, seed: int = 0) -> np.ndarray:
        """
        Forward simulation from each origin row: returns draws [len(origins), nsim, N]
        of the counts at row origin + lead (requires origin + lead < T).
        """
        rng = np.random.default_rng(seed)
        p = self.params
        psi = float(torch.exp(p['log_psi']))
        W = self.weights(p).numpy()
        origins = np.asarray(origins, dtype=int)
        steps = np.arange(1, lead + 1)
        out = np.empty((len(origins), nsim, self.N))
        with torch.no_grad():
            for k, o in enumerate(origins):
                log_nu, log_lam, log_phi = (r.numpy() for r in self.rates(p, torch.as_tensor(o + steps)))
                y = np.repeat(self.Y[o].numpy()[None, :], nsim, axis=0)
                for s in range(lead):
                    mu = np.exp(log_nu[s]) + np.exp(log_lam[s]) * y + np.exp(log_phi[s]) * (y @ W)
                    y = rng.poisson(rng.gamma(1.0 / psi, mu * psi))
                out[k] = y
        return out

    def scenario_params(self, scenario: str) -> dict:
        """
        Fitted parameters changed to a scenario with a different route of transmission:

        - ``fitted``: as estimated
        - ``no_ne``: no neighbourhood transmission (own rate as fitted)
        - ``strong_ne``: 75% of the (own + neighbour) rate via neighbours, same total
        - ``no_season``: no seasonality in the epidemic / neighbourhood rates
        """
        p = {k: v.clone() for k, v in self.params.items()}
        total = torch.exp(p['ar_int']) + torch.exp(p['ne_int'])
        if scenario == 'no_ne':
            p['ne_int'] = torch.full_like(p['ne_int'], -30.0)
        elif scenario == 'strong_ne':
            p['ar_int'] = torch.log(0.25 * total)
            p['ne_int'] = torch.log(0.75 * total)
        elif scenario == 'no_season':
            p['ar_seas'] = torch.zeros_like(p['ar_seas'])
            p['ne_seas'] = torch.zeros_like(p['ne_seas'])
        elif scenario != 'fitted':
            raise ValueError(f'unknown scenario {scenario!r}')
        return p

    def simulate_series(self, scenario: str = 'fitted', seed: int = 0, max_count: float = 1e7):
        """
        Simulate a whole new series (rows 1..T-1, starting from the observed row 0)
        under a scenario, and return (counts [T, N], true one-step components
        DataFrame with row, node, endemic, epidemic, neighbourhood, mean).
        Returns (None, None) if the simulated process explodes.
        """
        rng = np.random.default_rng(seed)
        p = self.scenario_params(scenario)
        psi = float(torch.exp(p['log_psi']))
        W = self.weights(p).numpy()
        rows = torch.arange(1, self.T)
        with torch.no_grad():
            log_nu, log_lam, log_phi = (r.numpy() for r in self.rates(p, rows))
        y = np.zeros((self.T, self.N))
        y[0] = self.Y[0].numpy()
        comp = np.zeros((3, self.T - 1, self.N))
        for k in range(self.T - 1):
            e = np.exp(log_nu[k]); a = np.exp(log_lam[k]) * y[k]; n = np.exp(log_phi[k]) * (y[k] @ W)
            comp[:, k] = e, a, n
            y[k + 1] = rng.poisson(rng.gamma(1.0 / psi, (e + a + n) * psi))
            if y[k + 1].max() > max_count:
                return None, None
        R = self.T - 1
        df = pd.DataFrame({'row': np.repeat(np.arange(1, self.T), self.N), 'node': np.tile(np.arange(self.N), R),
                           'endemic': comp[0].ravel(), 'epidemic': comp[1].ravel(),
                           'neighbourhood': comp[2].ravel()})
        df['mean'] = df[['endemic', 'epidemic', 'neighbourhood']].sum(axis=1)
        return y, df

    def node_parameters(self) -> pd.DataFrame:
        """Per region: endemic level, epidemic and neighbourhood rate at their seasonal mean."""
        p = self.params
        return pd.DataFrame({
            'node': np.arange(self.N),
            'endemic_effect': (p['end_int'] + p['b_end']).numpy(),
            'epidemic_rate': torch.exp(p['ar_int'] + p['b_ar']).numpy(),
            'neighbourhood_rate': torch.exp(p['ne_int'] + p['b_ne']).numpy(),
        })

    def summary(self) -> dict:
        p, f = self.params, self.fit_
        out = {'loglik': f.loglik, 'converged': f.converged, 'outer_iterations': f.n_iter,
               'overdispersion_psi': float(torch.exp(p['log_psi'])),
               'ar_intercept': float(p['ar_int']), 'ne_intercept': float(p['ne_int']),
               'end_intercept': float(p['end_int'])}
        if self.power_law:
            out['power_law_d'] = float(torch.exp(p['log_d']))
        out.update({f'sigma2_{k}': v for k, v in f.sigma2.items()})
        return out
