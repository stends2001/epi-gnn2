from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops


class HHH4Module(nn.Module):
    """
    hhh4-style GNN: three non-negative branches that add up to the NB mean.

    Mirrors the endemic-epidemic decomposition of hhh4 (Held, Hoehle & Hofmann
    2005; Meyer, Held & Hoehle 2017):

        mu_it = endemic_it + epidemic_it + neighbourhood_it
        Y_it  ~ NB(mu_it, alpha),    Var = mu + alpha * mu^2   (NB2)

    =============  ===========================================  =======================
    Branch         Reads                                        Mechanism
    =============  ===========================================  =======================
    endemic        seasonal / covariate features over window    MLP -> horizon
    epidemic       the node's own incidence history             linear -> horizon
    neighbourhood  neighbours' incidence history                GCN (no self-loops)
    =============  ===========================================  =======================

    Each branch ends in a softplus, so every contribution is >= 0 and the shares
    ``component / mu`` are interpretable.

    Only the total count is observed, so only the dispersion of the total is
    identifiable: there is ONE NB likelihood on ``mu``, not one per branch. Use
    ``sample_components`` for branch-level predictive draws that add up to the
    total.

    Parameters
    ----------
    num_nodes, seq_length, horizon_size : int
        Data dimensions.
    incidence_idx : list[int]
        Feature-axis indices of the incidence (lag) columns. With ``lag_num > 1``
        there are several. Order follows ``ColumnRegistry`` insertion order.
    endemic_idx : list[int]
        Feature-axis indices of the seasonal / covariate columns. May be empty, in
        which case the endemic branch is a learned per-horizon constant.
    hidden_size : int
        Width of the endemic MLP and of the neighbourhood embedding.
    num_layers : int
        Number of GCN layers in the neighbourhood branch. Keep this at 1 for a
        strict neighbourhood component: with 2+ layers a node's own signal can
        come back through 2-hop paths (i -> j -> i), even without self-loops.
    dropout_p : float
        Dropout in the endemic MLP and neighbourhood branch.
    norm_edges : bool
        Symmetric GCN normalisation of the edge weights.
    alpha_mode : {'global', 'node'}
        One dispersion for all nodes, or one per node.
    mu_init : float | None
        Expected mean count per node and step (e.g. the train mean). Used to set
        the output biases so that each branch starts at ``mu_init / 3``, instead
        of at softplus(0) = 0.69, which is far off for counts in the hundreds.

    Shapes
    ------
    x           : [num_nodes, num_features, seq_length]
    edge_index  : [2, num_edges]
    edge_weight : [num_edges] or None
    returns     : (mu [N, H], alpha [N, H]), and with ``return_components=True``
                  also a dict {'endemic', 'epidemic', 'neighbourhood'} of [N, H].
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
                 mu_init:       float | None = None):
        super().__init__()

        if len(incidence_idx) == 0:
            raise ValueError('HHH4Module needs at least one incidence feature index.')
        if set(incidence_idx) & set(endemic_idx):
            raise ValueError('incidence_idx and endemic_idx must be disjoint, '
                             'otherwise the branches are not separable.')
        if alpha_mode not in ('global', 'node'):
            raise ValueError(f"alpha_mode must be 'global' or 'node', got {alpha_mode!r}")

        self.num_nodes    = num_nodes
        self.seq_length   = seq_length
        self.horizon_size = horizon_size
        self.num_layers   = num_layers
        self.alpha_mode   = alpha_mode

        self.register_buffer('incidence_idx', torch.tensor(incidence_idx, dtype=torch.long))
        self.register_buffer('endemic_idx',   torch.tensor(endemic_idx,   dtype=torch.long))

        n_inc = len(incidence_idx) * seq_length
        n_end = len(endemic_idx) * seq_length

        # ---- endemic: covariates -> horizon ----
        if n_end > 0:
            self.endemic = nn.Sequential(
                nn.Linear(n_end, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_size, horizon_size),
            )
        else:
            self.endemic = None
            self.endemic_const = nn.Parameter(torch.zeros(horizon_size))

        # ---- epidemic: own history -> horizon (autoregressive) ----
        self.epidemic = nn.Linear(n_inc, horizon_size)

        # ---- neighbourhood: embed history, aggregate over neighbours only ----
        self.ne_embed = nn.Linear(n_inc, hidden_size)
        self.ne_convs = nn.ModuleList([
            GCNConv(hidden_size, hidden_size, add_self_loops=False, normalize=norm_edges)
            for _ in range(num_layers)
        ])
        self.ne_dropout = nn.Dropout(dropout_p)
        self.ne_out     = nn.Linear(hidden_size, horizon_size)

        # ---- dispersion ----
        n_alpha = num_nodes if alpha_mode == 'node' else 1
        self.log_alpha = nn.Parameter(torch.full((n_alpha,), math.log(0.1)))

        if mu_init is not None:
            self._init_output_bias(mu_init)

    # ------------------------------------------------------------------ #
    def _init_output_bias(self, mu_init: float) -> None:
        """Set output biases so that softplus(bias) = mu_init / 3 per branch."""
        target = max(float(mu_init) / 3.0, 1e-3)
        bias   = target + math.log(-math.expm1(-target))     # inverse softplus

        with torch.no_grad():
            if self.endemic is not None:
                self.endemic[-1].bias.fill_(bias)
            else:
                self.endemic_const.fill_(bias)
            self.epidemic.bias.fill_(bias)
            self.ne_out.bias.fill_(bias)

    def _select(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Feature subset, flattened over the window: [N, len(idx) * seq_length]."""
        # layout is (num_nodes, num_features, seq_length): index the FEATURE axis
        return x.index_select(1, idx).reshape(x.shape[0], -1)

    def forward(self,
                x:                 torch.Tensor,
                edge_index:        torch.Tensor,
                edge_weight:       torch.Tensor | None = None,
                return_components: bool = False):

        x_inc = self._select(x, self.incidence_idx)

        # endemic
        if self.endemic is not None:
            endemic = F.softplus(self.endemic(self._select(x, self.endemic_idx)))
        else:
            endemic = F.softplus(self.endemic_const).expand(x.shape[0], -1)

        # epidemic (own region)
        epidemic = F.softplus(self.epidemic(x_inc))

        # neighbourhood: strip any self-loops already in the graph. GCNConv's
        # add_self_loops=False only stops it from ADDING loops, it does not remove
        # existing ones, which would leak the node's own incidence into this branch.
        ei, ew = remove_self_loops(edge_index, edge_weight)
        h = F.relu(self.ne_embed(x_inc))
        for conv in self.ne_convs:
            h = F.relu(conv(h, ei, ew))
            h = self.ne_dropout(h)
        neighbourhood = F.softplus(self.ne_out(h))

        mu = endemic + epidemic + neighbourhood

        alpha = torch.exp(self.log_alpha)
        alpha = alpha.view(-1, 1).expand_as(mu) if self.alpha_mode == 'node' else alpha.expand_as(mu)

        if return_components:
            components = {'endemic': endemic, 'epidemic': epidemic, 'neighbourhood': neighbourhood}
            return (mu, alpha), components
        return mu, alpha


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
