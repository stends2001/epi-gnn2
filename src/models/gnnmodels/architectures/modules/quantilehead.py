from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneQuantileHead(nn.Module):
    """
    Output head that returns non-crossing quantiles.

    The head predicts the median directly, and the distance to each further
    quantile as a cumulative sum of softplus increments, upwards for the upper
    quantiles and downwards for the lower ones. Quantiles therefore come out
    sorted by construction, without post-hoc sorting (which would distort the
    pinball gradients).

    Parameters
    ----------
    in_features : int
        Size of the node embedding fed into the head.
    horizon_size : int
        Number of forecast steps.
    quantiles : list[float]
        Odd-length, sorted quantile levels with 0.5 in the middle (as validated
        by ``EpiConfig``).
    min_gap : float
        Small constant added to every increment, so neighbouring quantiles never
        coincide exactly.

    Shapes
    ------
    input  : [num_nodes, in_features]
    output : [num_nodes, horizon_size, num_quantiles]
    """
    def __init__(self,
                 in_features:  int,
                 horizon_size: int,
                 quantiles:    list[float],
                 min_gap:      float = 1e-4):
        super().__init__()

        num_q = len(quantiles)
        if num_q % 2 == 0 or abs(quantiles[num_q // 2] - 0.5) > 1e-9:
            raise ValueError(f'quantiles must be odd-length with 0.5 in the middle, got {quantiles}')

        self.horizon_size = horizon_size
        self.num_quantiles= num_q
        self.num_side     = num_q // 2
        self.min_gap      = min_gap

        self.median = nn.Linear(in_features, horizon_size)

        if self.num_side > 0:
            self.upper = nn.Linear(in_features, horizon_size * self.num_side)
            self.lower = nn.Linear(in_features, horizon_size * self.num_side)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        median = self.median(h)                                         # [N, H]

        if self.num_side == 0:
            return median.unsqueeze(-1)

        up   = F.softplus(self.upper(h)).view(n, self.horizon_size, self.num_side) + self.min_gap
        down = F.softplus(self.lower(h)).view(n, self.horizon_size, self.num_side) + self.min_gap

        upper_q = median.unsqueeze(-1) + torch.cumsum(up, dim=-1)       # q > 0.5, increasing
        lower_q = median.unsqueeze(-1) - torch.cumsum(down, dim=-1)     # q < 0.5, decreasing outward

        # order: lowest quantile first
        return torch.cat([lower_q.flip(-1), median.unsqueeze(-1), upper_q], dim=-1)
