from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, scatter

EndemicMode       = Literal['loglinear', 'mlp']
EpidemicMode      = Literal['rate', 'neural']
NeighbourhoodMode = Literal['rate', 'linear', 'gcn']


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) = y (y > 0)."""
    y = max(float(y), 1e-6)
    return y + math.log(-math.expm1(-y))


class HHH4Module(nn.Module):
    """
    hhh4-style GNN: three non-negative branches that add up to the NB mean.

    Mirrors the endemic-epidemic decomposition of hhh4 (Held, Hoehle & Hofmann
    2005; Meyer, Held & Hoehle 2017):

        mu_it = endemic_it + epidemic_it + neighbourhood_it
        Y_it  ~ NB(mu_it, alpha),    Var = mu + alpha * mu^2   (NB2)

    Default ("rate") form, as in hhh4::

        endemic_ih       = exp(a_i + b_h + sum_f (w_fh + d_ifh) x_f(t0))
        epidemic_ih      = exp(l_h + e_i) * sum_k pi_hk  y_i(t0 - k)
        neighbourhood_ih = exp(p_h + n_i) * sum_k rho_hk ybar_i(t0 - k)

    - ``a_i``: node intercept (initialised at the node's train mean), ``w``: shared
      seasonal / covariate coefficients, ``d_i``: node deviations. With week-of-year
      sin/cos features every region gets its own seasonal amplitude and peak week.
    - ``exp(l_h)`` / ``exp(p_h)``: epidemic and neighbourhood rates, ``e_i`` / ``n_i``:
      node effects on those rates.
    - ``pi``, ``rho``: lag weights over the input window (softmax, sum to 1).
    - ``ybar_i``: edge-weighted mean of the neighbours' counts (self-loops removed).

    Node deviations are centred over nodes and ridge-penalised (``regularization``),
    which pools regions with little data towards the shared values, like hhh4's
    random effects.

    Why rates: the neural alternative (softplus of a linear map, or GCN layers;
    ``epidemic_mode='neural'``, ``neighbourhood_mode='linear'|'gcn'``) can let one
    branch collapse to a zero share early in training. On simulated data with real
    spread between regions the neural forms ended near zero neighbourhood share
    with a worse fit; the rate form is linear in the counts, as the data-generating
    process is, and has interpretable parameters.

    Inputs of the epidemic / neighbourhood branches must be NON-NEGATIVE counts or
    rates (raw scale) in the rate form; they are divided by ``scale`` (train mean of
    the target) internally and clamped at 0.

    Parameters
    ----------
    num_nodes, seq_length, horizon_size : int
        Data dimensions.
    incidence_idx : list[int]
        Feature-axis indices of the case / incidence lag columns.
    endemic_idx : list[int]
        Feature-axis indices of the seasonal / covariate columns. May be empty.
    hidden_size, num_layers, dropout_p, norm_edges
        Only used by the neural forms (``'mlp'`` endemic, ``'neural'`` epidemic,
        ``'gcn'`` neighbourhood).
    alpha_mode : {'global', 'node'}
        One dispersion for all nodes, or one per node.
    mu_init : float | None
        Train mean of the target; sets ``scale``.
    node_means : array-like [num_nodes] | None
        Train mean per node; initialises the endemic intercepts.
    endemic_mode : {'loglinear', 'mlp'}
    epidemic_mode : {'rate', 'neural'}
    neighbourhood_mode : {'rate', 'linear', 'gcn'}
    node_effects : bool
        Node effects on the epidemic and neighbourhood rates.
    node_penalty : float
        Ridge penalty on the centred node deviations (mean of squares).

    Shapes
    ------
    x           : [num_nodes, num_features, seq_length]
    edge_index  : [2, num_edges]
    edge_weight : [num_edges] or None
    returns     : (mu [N, H], alpha [N, H]); with ``return_components=True`` also a
                  dict {'endemic', 'epidemic', 'neighbourhood'} of [N, H].
    """
    component_names = ('endemic', 'epidemic', 'neighbourhood')

    def __init__(self,
                 num_nodes:     int,
                 seq_length:    int,
                 horizon_size:  int,
                 incidence_idx: list[int],
                 endemic_idx:   list[int],
                 hidden_size:   int   = 32,
                 num_layers:    int   = 1,
                 dropout_p:     float = 0.1,
                 norm_edges:    bool  = True,
                 alpha_mode:    str   = 'global',
                 mu_init:       float | None = None,
                 node_means                  = None,
                 endemic_mode:  EndemicMode  = 'loglinear',
                 node_effects:  bool  = True,
                 node_penalty:  float = 0.01,
                 neighbourhood_mode: NeighbourhoodMode = 'rate',
                 epidemic_mode: EpidemicMode = 'rate'):
        super().__init__()

        if len(incidence_idx) == 0:
            raise ValueError('HHH4Module needs at least one incidence feature index.')
        if set(incidence_idx) & set(endemic_idx):
            raise ValueError('incidence_idx and endemic_idx must be disjoint, '
                             'otherwise the branches are not separable.')
        for name, val, ok in [('alpha_mode', alpha_mode, ('global', 'node')),
                              ('endemic_mode', endemic_mode, ('loglinear', 'mlp')),
                              ('epidemic_mode', epidemic_mode, ('rate', 'neural')),
                              ('neighbourhood_mode', neighbourhood_mode, ('rate', 'linear', 'gcn'))]:
            if val not in ok:
                raise ValueError(f'{name} must be one of {ok}, got {val!r}')

        self.num_nodes    = num_nodes
        self.seq_length   = seq_length
        self.horizon_size = horizon_size
        self.num_layers   = num_layers
        self.alpha_mode   = alpha_mode
        self.endemic_mode = endemic_mode
        self.epidemic_mode = epidemic_mode
        self.neighbourhood_mode = neighbourhood_mode
        self.node_effects = node_effects
        self.node_penalty = node_penalty

        scale = float(mu_init) if mu_init is not None and mu_init > 0 else 1.0
        self.register_buffer('scale',         torch.tensor(scale))
        self.register_buffer('incidence_idx', torch.tensor(incidence_idx, dtype=torch.long))
        self.register_buffer('endemic_idx',   torch.tensor(endemic_idx,   dtype=torch.long))

        n_inc = len(incidence_idx) * seq_length
        n_end = len(endemic_idx)
        third = 1.0 / 3.0                       # each branch starts at a third of the mean

        # ---- endemic ----
        if endemic_mode == 'loglinear':
            if node_means is not None:
                nm = np.clip(np.asarray(node_means, dtype=float), 1e-3 * scale, None)
                a0 = torch.tensor(np.log(nm * third / scale), dtype=torch.float32)
            else:
                a0 = torch.full((num_nodes,), math.log(third))
            self.end_node_intercept = nn.Parameter(a0)                                   # a_i
            self.end_horizon_bias   = nn.Parameter(torch.zeros(horizon_size))            # b_h
            self.end_coef           = nn.Parameter(torch.zeros(n_end, horizon_size))     # w_fh
            self.end_node_coef      = nn.Parameter(torch.zeros(num_nodes, n_end, horizon_size))  # d_ifh
        else:
            if n_end > 0:
                self.endemic = nn.Sequential(
                    nn.Linear(n_end * seq_length, hidden_size), nn.ReLU(),
                    nn.Dropout(dropout_p), nn.Linear(hidden_size, horizon_size))
                with torch.no_grad():
                    self.endemic[-1].bias.fill_(_inv_softplus(third))
            else:
                self.endemic = None
                self.endemic_const = nn.Parameter(torch.full((horizon_size,), _inv_softplus(third)))

        # ---- epidemic ----
        if epidemic_mode == 'rate':
            self.epi_log_rate   = nn.Parameter(torch.full((horizon_size,), math.log(third)))   # l_h
            self.epi_lag_logits = nn.Parameter(torch.zeros(horizon_size, n_inc))               # pi_h
        else:
            self.epidemic = nn.Linear(n_inc, horizon_size)
            with torch.no_grad():
                self.epidemic.bias.fill_(_inv_softplus(third))

        # ---- neighbourhood ----
        if neighbourhood_mode == 'rate':
            self.ne_log_rate   = nn.Parameter(torch.full((horizon_size,), math.log(third)))    # p_h
            self.ne_lag_logits = nn.Parameter(torch.zeros(horizon_size, n_inc))                # rho_h
        elif neighbourhood_mode == 'linear':
            self.ne_out = nn.Linear(n_inc, horizon_size)
            with torch.no_grad():
                self.ne_out.bias.fill_(_inv_softplus(third))
        else:
            self.ne_embed = nn.Linear(n_inc, hidden_size)
            self.ne_convs = nn.ModuleList([
                GCNConv(hidden_size, hidden_size, add_self_loops=False, normalize=norm_edges)
                for _ in range(num_layers)
            ])
            self.ne_dropout = nn.Dropout(dropout_p)
            self.ne_out     = nn.Linear(hidden_size, horizon_size)
            with torch.no_grad():
                self.ne_out.bias.fill_(_inv_softplus(third))

        # ---- node effects on epidemic / neighbourhood ----
        if node_effects:
            self.epi_node_effect = nn.Parameter(torch.zeros(num_nodes))   # e_i
            self.ne_node_effect  = nn.Parameter(torch.zeros(num_nodes))   # n_i

        # ---- dispersion ----
        n_alpha = num_nodes if alpha_mode == 'node' else 1
        self.log_alpha = nn.Parameter(torch.full((n_alpha,), math.log(0.1)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _centered(p: torch.Tensor) -> torch.Tensor:
        """
        Node deviations centred over nodes (dim 0). Without this, a common shift of
        all node effects would duplicate the shared parameters, and the 'node
        effect' would silently become a global scale (e.g. every multiplier 0.1).
        Centred, exp(effect) has geometric mean 1 across nodes: a node's value
        relative to the typical node.
        """
        return p - p.mean(dim=0, keepdim=True)

    @staticmethod
    def _neighbour_mean(h: torch.Tensor, edge_index: torch.Tensor,
                        edge_weight: torch.Tensor | None) -> torch.Tensor:
        """
        Edge-weighted mean of the source nodes' features per target node,
        sum_j w_ji h_j / sum_j w_ji. Nodes without neighbours get 0.
        """
        n = h.shape[0]
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index[0], edge_index[1]
        w = edge_weight if edge_weight is not None else torch.ones(src.shape[0], device=h.device)
        w = w.to(h.dtype)
        num = scatter(h[src] * w.unsqueeze(-1), dst, dim=0, dim_size=n, reduce='sum')
        den = scatter(w, dst, dim=0, dim_size=n, reduce='sum').clamp(min=1e-12).unsqueeze(-1)
        return num / den

    def _select(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Feature subset, flattened over the window: [N, len(idx) * seq_length]."""
        # layout is (num_nodes, num_features, seq_length): index the FEATURE axis
        return x.index_select(1, idx).reshape(x.shape[0], -1)

    @staticmethod
    def _lagged(h: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Lag-weighted average per horizon: [N, K] x softmax([H, K]) -> [N, H]."""
        return h @ torch.softmax(logits, dim=-1).t()

    def _node_mult(self, p: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._centered(p)).view(-1, 1)

    def _endemic(self, x: torch.Tensor) -> torch.Tensor:
        """Endemic rate in units of ``scale``: [N, H]."""
        n = x.shape[0]
        if self.endemic_mode == 'loglinear':
            log_rate = self.end_node_intercept.view(n, 1) + self.end_horizon_bias.view(1, -1)
            if self.endemic_idx.numel() > 0:
                z = x.index_select(1, self.endemic_idx)[:, :, -1]                    # [N, F] at t0
                coef = self.end_coef.unsqueeze(0) + self._centered(self.end_node_coef)   # [N, F, H]
                log_rate = log_rate + torch.einsum('nf,nfh->nh', z, coef)
            return torch.exp(log_rate.clamp(max=20.0))

        if self.endemic is not None:
            return F.softplus(self.endemic(self._select(x, self.endemic_idx)))
        return F.softplus(self.endemic_const).expand(n, -1)

    def forward(self,
                x:                 torch.Tensor,
                edge_index:        torch.Tensor,
                edge_weight:       torch.Tensor | None = None,
                return_components: bool = False):

        scale = self.scale
        x_inc = self._select(x, self.incidence_idx) / scale

        endemic = self._endemic(x) * scale

        # ---- epidemic (own region) ----
        if self.epidemic_mode == 'rate':
            own = self._lagged(x_inc.clamp(min=0), self.epi_lag_logits)
            epidemic = torch.exp(self.epi_log_rate).view(1, -1) * own * scale
        else:
            epidemic = F.softplus(self.epidemic(x_inc)) * scale

        # ---- neighbourhood ----
        # strip any self-loops already in the graph: GCNConv's add_self_loops=False only
        # stops it from ADDING loops, and a weighted mean over a self-loop would also leak
        # the node's own incidence into this branch.
        ei, ew = remove_self_loops(edge_index, edge_weight)
        if self.neighbourhood_mode == 'rate':
            nb = self._lagged(self._neighbour_mean(x_inc.clamp(min=0), ei, ew), self.ne_lag_logits)
            neighbourhood = torch.exp(self.ne_log_rate).view(1, -1) * nb * scale
        elif self.neighbourhood_mode == 'linear':
            neighbourhood = F.softplus(self.ne_out(self._neighbour_mean(x_inc, ei, ew))) * scale
        else:
            h = F.relu(self.ne_embed(x_inc))
            for conv in self.ne_convs:
                h = F.relu(conv(h, ei, ew))
                h = self.ne_dropout(h)
            neighbourhood = F.softplus(self.ne_out(h)) * scale

        if self.node_effects:
            epidemic      = epidemic      * self._node_mult(self.epi_node_effect)
            neighbourhood = neighbourhood * self._node_mult(self.ne_node_effect)

        mu = endemic + epidemic + neighbourhood

        alpha = torch.exp(self.log_alpha)
        alpha = alpha.view(-1, 1).expand_as(mu) if self.alpha_mode == 'node' else alpha.expand_as(mu)

        if return_components:
            components = {'endemic': endemic, 'epidemic': epidemic, 'neighbourhood': neighbourhood}
            return (mu, alpha), components
        return mu, alpha

    # ------------------------------------------------------------------ #
    def regularization(self) -> torch.Tensor:
        """Ridge penalty pulling centred node deviations towards the shared parameters."""
        terms = []
        if self.endemic_mode == 'loglinear' and self.end_node_coef.numel() > 0:
            terms.append(self._centered(self.end_node_coef).pow(2).mean())
        if self.node_effects:
            terms.append(self._centered(self.epi_node_effect).pow(2).mean())
            terms.append(self._centered(self.ne_node_effect).pow(2).mean())
        if not terms or self.node_penalty == 0:
            return self.log_alpha.new_zeros(())
        return self.node_penalty * torch.stack(terms).sum()

    @torch.no_grad()
    def node_parameters(self) -> dict[str, np.ndarray]:
        """
        Learned parameters as numpy arrays.

        - ``endemic_baseline`` [N, H]: endemic level with all endemic features at 0,
          in target units (counts or rates).
        - ``endemic_coef`` [N, F, H]: total coefficient per endemic feature
          (shared + centred node deviation), log scale. Log-linear mode only.
        - ``epidemic_rate`` / ``neighbourhood_rate`` [N, H]: expected new cases per
          (lag-weighted) case in the own region / per neighbour-mean case. Rate
          form only; includes the node effects.
        - ``epidemic_lag_weights`` / ``neighbourhood_lag_weights`` [H, K]: weights
          over the input window columns (rate form only).
        - ``epidemic_multiplier`` / ``neighbourhood_multiplier`` [N]: node effects,
          centred so their geometric mean over nodes is 1.
        - ``alpha`` [N]: NB dispersion per node (constant if global).
        """
        out: dict[str, np.ndarray] = {}
        N = self.num_nodes
        ones = torch.ones(N, device=self.log_alpha.device)
        epi_mult = torch.exp(self._centered(self.epi_node_effect)) if self.node_effects else ones
        ne_mult  = torch.exp(self._centered(self.ne_node_effect))  if self.node_effects else ones

        if self.endemic_mode == 'loglinear':
            base = torch.exp(self.end_node_intercept.view(N, 1) + self.end_horizon_bias.view(1, -1)) * self.scale
            out['endemic_baseline'] = base.cpu().numpy()
            out['endemic_coef']     = (self.end_coef.unsqueeze(0) + self._centered(self.end_node_coef)).cpu().numpy()

        if self.epidemic_mode == 'rate':
            out['epidemic_rate'] = (epi_mult.view(-1, 1) * torch.exp(self.epi_log_rate).view(1, -1)).cpu().numpy()
            out['epidemic_lag_weights'] = torch.softmax(self.epi_lag_logits, -1).cpu().numpy()
        if self.neighbourhood_mode == 'rate':
            out['neighbourhood_rate'] = (ne_mult.view(-1, 1) * torch.exp(self.ne_log_rate).view(1, -1)).cpu().numpy()
            out['neighbourhood_lag_weights'] = torch.softmax(self.ne_lag_logits, -1).cpu().numpy()

        out['epidemic_multiplier']      = epi_mult.cpu().numpy()
        out['neighbourhood_multiplier'] = ne_mult.cpu().numpy()
        out['alpha'] = torch.exp(self.log_alpha).expand(N).cpu().numpy()
        return out


# ====================================================================== #
# Distribution helpers (shared by HHH4Model and the forecasting mixin)
# ====================================================================== #
def nb_quantiles(mu, alpha, quantiles):
    """
    Exact NB2 quantiles via scipy, as a float numpy array [..., Q].

    Parameterisation: size r = 1/alpha, success probability p = r / (r + mu),
    so mean = mu and variance = mu + alpha * mu^2.
    """
    import numpy as np
    from scipy.stats import nbinom

    mu    = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
    alpha = np.clip(np.asarray(alpha, dtype=float), 1e-10, None)
    r     = 1.0 / alpha
    p     = r / (r + mu)

    q = np.asarray(quantiles, dtype=float)
    return nbinom.ppf(q, r[..., None], p[..., None]).astype(float)


def sample_components(mu_components: dict[str, torch.Tensor],
                      alpha:         torch.Tensor,
                      n_samples:     int = 1000,
                      generator:     torch.Generator | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Joint predictive draws for the total and each branch.

    1. draw the total from the Gamma-Poisson mixture: lambda ~ Gamma(1/alpha, rate=1/(alpha*mu)),
       Y ~ Poisson(lambda), so Y ~ NB(mu, alpha);
    2. split each draw multinomially over the branches with probabilities mu_k / mu.

    The branch draws add up exactly to the total draw, and they need no extra
    (unidentifiable) per-branch dispersion. They are model-implied attributions,
    not forecasts that can be checked for calibration: the components are never
    observed.

    Returns
    -------
    total : [n_samples, *mu.shape]
    parts : dict of [n_samples, *mu.shape], same keys as ``mu_components``
    """
    names = list(mu_components.keys())
    stack = torch.stack([mu_components[k] for k in names], dim=-1).clamp(min=0)   # [..., K]
    mu    = stack.sum(-1).clamp(min=1e-10)
    alpha = alpha.expand_as(mu).clamp(min=1e-10)

    shape = 1.0 / alpha
    rate  = 1.0 / (alpha * mu)
    gamma = torch.distributions.Gamma(shape, rate)
    lam   = gamma.sample((n_samples,))
    total = torch.poisson(lam, generator=generator)                                 # [S, ...]

    probs = (stack / mu.unsqueeze(-1)).expand(n_samples, *stack.shape)              # [S, ..., K]
    flat_p = probs.reshape(-1, len(names))
    flat_n = total.reshape(-1)

    # multinomial thinning via sequential binomials (vectorised, exact)
    parts_flat = []
    remaining  = flat_n.clone()
    remaining_p = torch.ones_like(flat_n)
    for k in range(len(names) - 1):
        cond_p = (flat_p[:, k] / remaining_p.clamp(min=1e-12)).clamp(0, 1)
        draw   = torch.binomial(remaining, cond_p, generator=generator)
        parts_flat.append(draw)
        remaining   = remaining - draw
        remaining_p = remaining_p - flat_p[:, k]
    parts_flat.append(remaining)

    parts = {k: v.reshape(total.shape) for k, v in zip(names, parts_flat)}
    return total, parts
