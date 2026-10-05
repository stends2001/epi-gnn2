from __future__ import annotations

import torch

from .baseloss import BaseLoss


class PinballLoss(BaseLoss):
    """
    Pinball (quantile) loss, averaged over quantile levels.

    For a quantile level ``q`` and error ``e = y - y_hat_q``, the loss is
    ``max(q * e, (q - 1) * e)``. Minimizing it makes ``y_hat_q`` the ``q``-quantile
    of the predictive distribution. Averaged over levels, it equals (up to a
    constant) the quantile score that WIS approximates.

    Registered as ``'pinball'`` in ``LossManager``.

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels, in the same order as the last axis of ``y_pred``.

    Shapes
    ------
    y_pred : [num_nodes, horizon_size, num_quantiles]
    y_true : [num_nodes, horizon_size]
    """
    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=list(quantiles))
        self.register_buffer('levels', torch.tensor(list(quantiles), dtype=torch.float32))

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() != y_true.dim() + 1 or y_pred.shape[-1] != self.levels.numel():
            raise ValueError(
                f'PinballLoss expects y_pred [..., {self.levels.numel()}] and y_true [...]; '
                f'got {tuple(y_pred.shape)} and {tuple(y_true.shape)}'
            )
        levels = self.levels.to(y_pred.device, y_pred.dtype)
        error  = y_true.unsqueeze(-1) - y_pred
        loss   = torch.maximum(levels * error, (levels - 1.0) * error)
        return loss.mean()
