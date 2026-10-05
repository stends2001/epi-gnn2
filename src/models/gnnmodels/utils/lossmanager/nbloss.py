from .baseloss import BaseLoss
import torch
import torch.nn as nn

class NBLoss(BaseLoss):
    """
    NB - Loss

    Negative Binomial negative log-likelihood loss, for use with
    `HHH4Module`'s (mu, alpha) output in place of an MSE-based loss.
    Uses the NB2 mean-dispersion parameterization common in
    epidemiological count regression (Cameron & Trivedi, 2013), where
    the predicted mean is `mu` and the variance is `mu + alpha * mu^2`.
    As `alpha -> 0`, the NB likelihood converges to a Poisson
    likelihood; `alpha` therefore has a direct epidemiological
    reading as an overdispersion parameter.

    Parameters
    ----------
    reduction: str
        one of {'mean', 'sum', 'none'}, matching `torch.nn` convention
    eps: float
        small constant added to `mu` and `alpha` for numerical
        stability when either is driven close to zero

    Forward
    -------
    Given target counts `y`, predicted mean `mu` and predicted
    dispersion `alpha` (all broadcastable to the same shape), computes
    the NB2 negative log-likelihood:

        log p(y | mu, alpha) =
              lgamma(y + 1/alpha) - lgamma(y + 1) - lgamma(1/alpha)
            + (1/alpha) * log(1 / (1 + alpha * mu))
            + y * log(alpha * mu / (1 + alpha * mu))

    and returns its negative, reduced according to `reduction`.
    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.reduction = reduction
        self.eps       = eps

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mu, alpha = y_pred[0], y_pred[1]

        mu    = mu.clamp(min=self.eps)
        alpha = alpha.clamp(min=self.eps)
        alpha = alpha.expand_as(mu) if alpha.shape != mu.shape else alpha

        r = 1.0 / alpha  # inverse-dispersion, a.k.a. the NB "size" parameter

        log_lik = (
            torch.lgamma(y_true + r)
            - torch.lgamma(y_true + 1.0)
            - torch.lgamma(r)
            + r * torch.log(r / (r + mu))
            + y_true * torch.log(mu / (r + mu))
        )
        nll = -log_lik

        if self.reduction == 'mean':
            return nll.mean()
        if self.reduction == 'sum':
            return nll.sum()
        return nll

    @staticmethod
    def predictive_interval(mu: torch.Tensor, alpha: torch.Tensor,
                             lower_q: float = 0.025, upper_q: float = 0.975,
                             n_samples: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte-Carlo prediction interval for the NB2 distribution defined
        by `(mu, alpha)`. For forecasts, ``hhh4module.nb_quantiles`` gives the
        exact quantiles (scipy ``nbinom.ppf``) and is what ``forecast()`` uses;
        this sampler is kept as a cross-check.

        Parameters
        ----------
        mu: torch.Tensor
            predicted NB mean, any shape
        alpha: torch.Tensor
            predicted NB dispersion, broadcastable to `mu`'s shape
        lower_q, upper_q: float
            quantiles defining the interval, e.g. 0.025/0.975 for a
            95% interval
        n_samples: int
            number of Monte Carlo draws per (mu, alpha) pair

        Returns
        -------
        lower, upper: torch.Tensor
            interval bounds, same shape as `mu`
        """
        mu    = mu.clamp(min=1e-8)
        alpha = alpha.clamp(min=1e-8)
        alpha = alpha.expand_as(mu) if alpha.shape != mu.shape else alpha

        r = 1.0 / alpha
        p = r / (r + mu)  # NB "success probability" in the (r, p) parameterization

        r_exp = r.unsqueeze(0).expand(n_samples, *r.shape)
        p_exp = p.unsqueeze(0).expand(n_samples, *p.shape)

        gamma_dist = torch.distributions.Gamma(concentration=r_exp, rate=p_exp / (1 - p_exp))
        lam = gamma_dist.sample()
        samples = torch.poisson(lam)

        lower = torch.quantile(samples, lower_q, dim=0)
        upper = torch.quantile(samples, upper_q, dim=0)
        return lower, upper
