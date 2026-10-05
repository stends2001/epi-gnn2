from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
import pandas as pd
import torch

from ...utils import Strategy
from ...gnnmodel import GNNModel
from .....dataloading import GraphDataBuilder
from .....utils import DataSetSplit

from ..modules.hhh4module import HHH4Module, sample_components


class HHH4Model(GNNModel):
    """
    GNN with an hhh4-style endemic / epidemic / neighbourhood decomposition and a
    Negative Binomial output. See ``HHH4Module`` for the architecture.

    Train with ``loss='nb'`` (the default picked by ``set_global_hparams``).

    Data requirements
    -----------------
    The NB likelihood acts on the raw, non-negative target. The target must
    therefore NOT be log-transformed or normalised (``normalization_method=None``
    and the target not in ``log_transform``). Feature columns may still be
    transformed. Counts (``target_column='cases'``) are the natural scale;
    incidence rates work as a quasi-likelihood, but NB quantiles are integers,
    so intervals on rates are coarse.

    Explanation
    -----------
    ``forecast_components(dataset)`` returns the per-branch means and shares.
    ``sample_component_draws(dataset)`` returns joint predictive draws per branch
    via multinomial thinning of the total.
    """
    _expected_databuilder = 'GraphDataBuilder'

    def __init__(self,
                 databuilder: GraphDataBuilder,
                 name:        str = 'hhh4model'):

        super().__init__(
            databuilder = databuilder,
            name        = name,
            strategy    = Strategy()
        )

    def set_model_hparams(self,
                          hidden_size:        int   = 32,
                          num_layers:         int   = 1,
                          dropout:            float = 0.1,
                          norm_edges:         bool  = True,
                          alpha_mode:         Literal['global', 'node'] = 'global',
                          incidence_features: list[str] | None = None,
                          endemic_features:   list[str] | None = None,
                          init_from_train:    bool  = True):
        """
        Parameters
        ----------
        hidden_size, num_layers, dropout, norm_edges, alpha_mode
            See ``HHH4Module``. ``num_layers=1`` keeps the neighbourhood branch
            strictly first-order.
        incidence_features : list[str] | None
            Feature columns read by the epidemic and neighbourhood branches.
            Default: all ``{lag_column}_lag*`` features.
        endemic_features : list[str] | None
            Feature columns read by the endemic branch. Default: every other
            feature (seasonal sin/cos encodings, population size/density).
        init_from_train : bool
            Initialise the output biases at the mean train target, so training
            starts at the right count scale.
        """
        self._set_output_head('nb')
        self._check_target_untransformed()

        feature_names = self.column_registration.get_entries_names_by_type('feature')
        lag_prefix    = f'{self.epiconfig.lag_column}_lag'

        if incidence_features is None:
            incidence_features = [f for f in feature_names if f.startswith(lag_prefix)]
        if endemic_features is None:
            endemic_features = [f for f in feature_names if f not in incidence_features]

        missing = [f for f in incidence_features + endemic_features if f not in feature_names]
        if missing:
            raise ValueError(f'Unknown feature columns {missing}. Available (in tensor order): {feature_names}')

        # indices follow ColumnRegistry insertion order = feature-axis order of x
        incidence_idx = [feature_names.index(f) for f in incidence_features]
        endemic_idx   = [feature_names.index(f) for f in endemic_features]

        self.incidence_features = list(incidence_features)
        self.endemic_features   = list(endemic_features)

        mu_init = self._mean_train_target() if init_from_train else None

        self.model = HHH4Module(
            num_nodes     = len(self.databuilder.dataorchestrator.data_context.local_shapedata),
            seq_length    = self.epiconfig.sequence_length,
            horizon_size  = self.epiconfig.horizon_size,
            incidence_idx = incidence_idx,
            endemic_idx   = endemic_idx,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout_p     = dropout,
            norm_edges    = norm_edges,
            alpha_mode    = alpha_mode,
            mu_init       = mu_init,
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'hidden_size':        hidden_size,
            'num_layers':         num_layers,
            'dropout':            dropout,
            'norm_edges':         norm_edges,
            'alpha_mode':         alpha_mode,
            'incidence_features': self.incidence_features,
            'endemic_features':   self.endemic_features,
            'init_from_train':    init_from_train,
        }

        self._update_status('model_hparams_set')

    # ======================================================================= #
    # explanation
    # ======================================================================= #
    def forecast_components(self, dataset: DataSetSplit = 'test') -> pd.DataFrame:
        """
        Per-branch NB means for every (t0, node, horizon), with their shares of mu.

        Columns: t0 timestamp, node, horizon, endemic, epidemic, neighbourhood,
        mu, alpha, share_endemic, share_epidemic, share_neighbourhood, target.
        """
        self._check_status(['model_hparams_set', 'trained'])
        comps, mu, alpha, y = self._collect_components(dataset)
        T, N, H = mu.shape

        t0 = self._t0_timestamps(dataset, T)
        frames = []
        for hh in range(H):
            df = pd.DataFrame({
                self.epiconfig.temporal_column: np.repeat(t0, N),
                self.epiconfig.id_column:       np.tile(np.arange(N), T),
                'horizon':                      hh,
            })
            for name in HHH4Module.component_names:
                df[name] = comps[name][:, :, hh].reshape(-1)
            df['mu']     = mu[:, :, hh].reshape(-1)
            df['alpha']  = alpha[:, :, hh].reshape(-1)
            for name in HHH4Module.component_names:
                df[f'share_{name}'] = df[name] / df['mu']
            df['target'] = y[:, :, hh].reshape(-1)
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def sample_component_draws(self,
                               dataset:   DataSetSplit = 'test',
                               n_samples: int = 1000,
                               seed:      int | None = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Joint predictive draws of the total and each branch, [S, T, N, H].

        Branch draws add up to the total draw (multinomial thinning), so branch
        intervals are consistent with the total interval. They are attributions
        under the model, not separately calibrated forecasts.
        """
        self._check_status(['model_hparams_set', 'trained'])
        comps, _, alpha, _ = self._collect_components(dataset)

        if seed is not None:
            torch.manual_seed(seed)

        total, parts = sample_components(
            {k: torch.as_tensor(v) for k, v in comps.items()},
            torch.as_tensor(alpha),
            n_samples = n_samples,
        )
        return total.numpy(), {k: v.numpy() for k, v in parts.items()}

    # ======================================================================= #
    # helpers
    # ======================================================================= #
    def _collect_components(self, dataset: DataSetSplit):
        """Run the model with return_components over a split; numpy arrays [T, N, H]."""
        self.model.eval()
        comps: dict[str, list[np.ndarray]] = {k: [] for k in HHH4Module.component_names}
        mus, alphas, ys = [], [], []

        with torch.no_grad():
            for snapshot in self._get_dataloader(dataset):
                snapshot = snapshot.to(self.device)
                assert snapshot.graph is not None
                (mu, alpha), parts = self.model(snapshot.x,
                                                snapshot.graph.edge_index,
                                                snapshot.graph.edge_weight,
                                                return_components=True)
                for k in comps:
                    comps[k].append(parts[k].cpu().numpy())
                mus.append(mu.cpu().numpy())
                alphas.append(alpha.cpu().numpy())
                ys.append(snapshot.y.cpu().numpy())

        return ({k: np.stack(v) for k, v in comps.items()},
                np.stack(mus), np.stack(alphas), np.stack(ys))

    def _check_target_untransformed(self) -> None:
        entry  = self.column_registration.get_entry_by_name('target')
        params = entry.transformation_params

        if params is not None and any(getattr(params, a, None) is not None for a in ('log', 'zscore', 'minmax')):
            raise ValueError(
                'HHH4Model fits an NB likelihood on the raw target, but the target is '
                'transformed (log / zscore / minmax). Set normalization_method=None and '
                'leave the target out of log_transform in EpiConfig.'
            )

        if self.epiconfig.target_column != 'cases':
            warnings.warn(
                f"HHH4Model on target '{self.epiconfig.target_column}': NB is a count "
                "distribution, so this is a quasi-likelihood fit and NB quantiles will be "
                "integers on the rate scale. Counts (or a population offset) are preferred.",
                stacklevel=2,
            )

    def _mean_train_target(self) -> float:
        total, count = 0.0, 0
        for snapshot in self.databuilder.dataloader_train:
            total += float(snapshot.y.sum())
            count += snapshot.y.numel()
        return total / max(count, 1)
