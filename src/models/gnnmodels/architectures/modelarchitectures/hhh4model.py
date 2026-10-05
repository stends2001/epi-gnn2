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
                          init_from_train:    bool  = True,
                          endemic_mode:       Literal['loglinear', 'mlp'] = 'loglinear',
                          node_effects:       bool  = True,
                          node_penalty:       float = 0.01,
                          neighbourhood_mode: Literal['rate', 'linear', 'gcn'] = 'rate',
                          epidemic_mode:      Literal['rate', 'neural'] = 'rate',
                          seasonal_rates:     bool  = True,
                          rate_dynamics:      Literal['none', 'gru', 'lstm'] = 'gru',
                          dynamics_hidden:    int   = 16,
                          max_log_rate_adj:   float = 3.0,
                          dynamics_penalty:   float = 1e-3,
                          disabled_branches:  list[str] | None = None):
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
            Scale the network by the mean train target and initialise each node's
            endemic level at its own train mean, so training starts at the right
            count scale.
        endemic_mode : {'loglinear', 'mlp'}
            ``'loglinear'``: hhh4-style node-specific seasonality (intercept, and
            per-node coefficients on the seasonal features). ``'mlp'``: the earlier
            shared MLP, kept for comparison.
        node_effects : bool
            Node multipliers on the epidemic and neighbourhood branches.
        node_penalty : float
            Ridge penalty pulling node-specific parameters towards the shared ones.
            Larger = more pooling; 0 = independent per node.
        neighbourhood_mode : {'rate', 'linear', 'gcn'}
            ``'rate'`` (default, as in hhh4): a rate times the lag-weighted,
            edge-weighted mean of the neighbours' counts. ``'linear'`` / ``'gcn'``:
            neural forms; more flexible but can collapse to a zero share.
        epidemic_mode : {'rate', 'neural'}
            ``'rate'`` (default): a rate times the lag-weighted own counts.
            ``'neural'``: softplus of a linear map (the first version).

        seasonal_rates : bool
            Seasonal (week-of-year) terms on the epidemic and neighbourhood rates.
        rate_dynamics : {'none', 'gru', 'lstm'}
            A recurrent unit reads the recent trajectory (own and neighbour counts,
            and their weekly change) and shifts both rates per region and week, so
            the model can follow fast growth and decline (influenza). ``'none'``
            gives constant rates (plus the seasonal terms).
        disabled_branches : list[str] | None
            Branches fixed at 0, for ablations.
        dynamics_hidden, max_log_rate_adj, dynamics_penalty
            Hidden size of the recurrent unit; bound on its log-rate shift
            (3.0 = rates scaled by up to e^3 = 20x either way); ridge penalty on it.

        The rate forms need non-negative, untransformed lag features: use
        ``target_column='cases', lag_column='cases'`` (case lags follow the raw
        target) or keep the incidence lags out of the log / normalisation.
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
        if 'rate' in (epidemic_mode, neighbourhood_mode):
            self._check_features_untransformed(incidence_features)
        endemic_idx   = [feature_names.index(f) for f in endemic_features]

        self.incidence_features = list(incidence_features)
        self.endemic_features   = list(endemic_features)

        mu_init, node_means = self._train_target_means() if init_from_train else (None, None)

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
            node_means    = node_means,
            endemic_mode  = endemic_mode,
            node_effects  = node_effects,
            node_penalty  = node_penalty,
            neighbourhood_mode = neighbourhood_mode,
            epidemic_mode = epidemic_mode,
            seasonal_rates = seasonal_rates,
            rate_dynamics = rate_dynamics,
            dynamics_hidden = dynamics_hidden,
            max_log_rate_adj = max_log_rate_adj,
            dynamics_penalty = dynamics_penalty,
            disabled_branches = tuple(disabled_branches or ()),
        ).to(self.device)
        self.alpha_scale = 1.0

        self.config_info['model_hparams'] = {
            'hidden_size':        hidden_size,
            'num_layers':         num_layers,
            'dropout':            dropout,
            'norm_edges':         norm_edges,
            'alpha_mode':         alpha_mode,
            'incidence_features': self.incidence_features,
            'endemic_features':   self.endemic_features,
            'init_from_train':    init_from_train,
            'endemic_mode':       endemic_mode,
            'node_effects':       node_effects,
            'node_penalty':       node_penalty,
            'neighbourhood_mode': neighbourhood_mode,
            'epidemic_mode':      epidemic_mode,
            'seasonal_rates':     seasonal_rates,
            'rate_dynamics':      rate_dynamics,
            'dynamics_hidden':    dynamics_hidden,
            'max_log_rate_adj':   max_log_rate_adj,
            'dynamics_penalty':   dynamics_penalty,
            'disabled_branches':  list(disabled_branches or []),
        }

        self._update_status('model_hparams_set')

    # ======================================================================= #
    # explanation
    # ======================================================================= #
    def forecast_components(self, dataset: DataSetSplit = 'test') -> pd.DataFrame:
        """
        Per-branch NB means for every (t0, node, horizon), with their shares of mu.

        Columns: t0 timestamp, node, horizon, endemic, epidemic, neighbourhood,
        mu, alpha, share_endemic, share_epidemic, share_neighbourhood, target, and
        for the rate forms the time-varying rate multipliers
        (epidemic_rate_multiplier, neighbourhood_rate_multiplier).
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
            mult = getattr(self, '_last_rate_multipliers', None)
            if mult is not None:
                # time-varying part of the rates (seasonal terms x recurrent shift);
                # > 1: the branch is amplified in that week, e.g. in a growth phase
                df['epidemic_rate_multiplier']      = mult[0][:, :, hh].reshape(-1)
                df['neighbourhood_rate_multiplier'] = mult[1][:, :, hh].reshape(-1)
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def predictive_nb(self, dataset: DataSetSplit = 'test', horizon: int = 0) -> pd.DataFrame:
        """
        NB predictive distribution per row: target_time, node, target, mu, alpha
        (dispersion including the calibration scale). Used for the randomised PIT.
        """
        comp = self.forecast_components(dataset)
        comp = comp[comp['horizon'] == horizon]
        steps = self.epiconfig.horizon_leadtime + horizon
        out = pd.DataFrame({
            'target_time': pd.to_datetime(comp[self.epiconfig.temporal_column]) + pd.Timedelta(weeks=steps),
            self.epiconfig.id_column: comp[self.epiconfig.id_column].to_numpy(),
            'target': comp['target'].to_numpy(), 'mu': comp['mu'].to_numpy(), 'alpha': comp['alpha'].to_numpy(),
        })
        return out.reset_index(drop=True)

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
    # interval calibration
    # ======================================================================= #
    def calibrate_dispersion(self,
                             dataset: DataSetSplit = 'val',
                             season: str | None = 'in',
                             scales=None) -> pd.DataFrame:
        """
        Choose one multiplier for the NB dispersion that minimises the WIS on
        ``dataset`` (default: in-season weeks of the validation split), and use it
        for all later forecasts.

        The dispersion is fitted by the likelihood over all weeks, including the
        noisy off-season, which can leave in-season intervals too wide (or too
        narrow). This rescales the spread only; the mean is unchanged. It is the
        NB counterpart of the conformal correction used for the other GNNs.
        Call it after ``train()`` and before ``forecast()``. Returns the WIS and
        coverage for every candidate multiplier.
        """
        from ....utils.intervalmetrics import wis, coverage_and_width, season_mask, season_weeks_for
        from ..modules.hhh4module import nb_quantiles

        self._check_status(['model_hparams_set', 'trained'])
        q = self.epiconfig.quantiles
        if q is None:
            raise ValueError('calibrate_dispersion needs interval mode (EpiConfig.quantiles).')

        old = getattr(self, 'alpha_scale', 1.0)
        self.alpha_scale = 1.0
        comp = self.forecast_components(dataset)
        self.alpha_scale = old

        if season is not None:
            steps = self.epiconfig.horizon_leadtime + comp['horizon']
            target_time = pd.to_datetime(comp[self.epiconfig.temporal_column]) + pd.to_timedelta(7 * steps, unit='D')
            mask = season_mask(target_time, *season_weeks_for(self.epiconfig.disease)).to_numpy()
            if season == 'off':
                mask = ~mask
            if mask.sum() > 0:
                comp = comp[mask]

        mu, alpha, y = comp['mu'].to_numpy(), comp['alpha'].to_numpy(), comp['target'].to_numpy()
        scales = np.exp(np.linspace(np.log(0.02), np.log(50.0), 41)) if scales is None else np.asarray(scales)

        rows = []
        for c in scales:
            qv = nb_quantiles(mu, alpha * c, q)
            df = pd.DataFrame({'target': y, **{f'pred_q{i+1}': qv[:, i] for i in range(len(q))}})
            cw = coverage_and_width(df, q)
            rows.append({'alpha_scale': float(c), 'wis': float(wis(df, q).mean()),
                         **{f"cov{int(round(n * 100))}": cv for n, cv in zip(cw['nominal'], cw['coverage'])}})
        table = pd.DataFrame(rows)
        self.alpha_scale = float(table.loc[table['wis'].idxmin(), 'alpha_scale'])
        self.config_info['alpha_scale'] = self.alpha_scale
        return table

    # ======================================================================= #
    # helpers
    # ======================================================================= #
    def _collect_components(self, dataset: DataSetSplit):
        """Run the model with return_components over a split; numpy arrays [T, N, H]."""
        self.model.eval()
        comps: dict[str, list[np.ndarray]] = {k: [] for k in HHH4Module.component_names}
        mus, alphas, ys = [], [], []
        epi_mult, ne_mult = [], []

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
                mult = getattr(self.model, 'last_rate_multipliers', None)
                if mult is not None:
                    epi_mult.append(mult[0].cpu().numpy())
                    ne_mult.append(mult[1].cpu().numpy())
                mus.append(mu.cpu().numpy())
                alphas.append(alpha.cpu().numpy() * getattr(self, 'alpha_scale', 1.0))
                ys.append(snapshot.y.cpu().numpy())

        self._last_rate_multipliers = (np.stack(epi_mult), np.stack(ne_mult)) if epi_mult else None
        return ({k: np.stack(v) for k, v in comps.items()},
                np.stack(mus), np.stack(alphas), np.stack(ys))

    def _check_features_untransformed(self, features: list[str]) -> None:
        reg = self.column_registration
        bad = []
        for f in features:
            entry = reg.get_entry_by_name(f)
            if not entry.transformation:
                continue
            group = entry.transformation_group
            params = entry.transformation_params if group == 'self' else (
                reg.get_entry_by_name(group).transformation_params if group else None)
            if params is not None and any(getattr(params, a, None) is not None
                                          for a in ('log', 'zscore', 'minmax')):
                bad.append(f)
        if bad:
            raise ValueError(
                f'The rate form needs raw, non-negative lag features, but {bad} are '
                "transformed. Use target_column='cases' with lag_column='cases' (case "
                "lags stay raw), or epidemic_mode='neural' / neighbourhood_mode='gcn'."
            )

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

    def _train_target_means(self) -> tuple[float, np.ndarray]:
        """Mean train target overall and per node."""
        total, count = None, 0
        for snapshot in self.databuilder.dataloader_train:
            y = snapshot.y.detach().cpu().double()
            total = y.sum(dim=1) if total is None else total + y.sum(dim=1)
            count += y.shape[1]
        if total is None:
            raise ValueError('Empty training dataloader.')
        node_means = (total / count).numpy()
        return float(node_means.mean()), node_means

    # ======================================================================= #
    # parameters
    # ======================================================================= #
    def node_parameters(self) -> pd.DataFrame:
        """
        Learned per-node parameters, one row per (node, horizon).

        Columns
        -------
        node, node_name, horizon,
        endemic_baseline   : endemic level with all endemic features at 0
        seasonal_amplitude : peak-to-trough ratio of the endemic seasonal curve
                             (exp(2A) for log-rate A*cos(...)); 1 = no seasonality
        endemic_peak_week  : week of year (target time) at which the endemic
                             curve peaks
        epidemic_rate      : expected cases per (lag-weighted) own case (rate form)
        neighbourhood_rate : expected cases per neighbour-mean case (rate form)
        epidemic_multiplier, neighbourhood_multiplier : node effects, geometric mean 1
        alpha              : NB dispersion

        Seasonal columns need the log-linear endemic and the week-of-year
        features (``time_index_w=True``).
        """
        p  = self.model.node_parameters()
        N  = self.model.num_nodes
        H  = self.epiconfig.horizon_size
        id_col = self.epiconfig.id_column

        rows = []
        sin_i = cos_i = None
        if 'endemic_coef' in p:
            names = self.endemic_features
            sin_i = next((i for i, f in enumerate(names) if f.endswith('sin_w')), None)
            cos_i = next((i for i, f in enumerate(names) if f.endswith('cos_w')), None)

        for hh in range(H):
            df = pd.DataFrame({id_col: np.arange(N), 'horizon': hh})
            if 'endemic_baseline' in p:
                df['endemic_baseline'] = p['endemic_baseline'][:, hh]
            if sin_i is not None and cos_i is not None:
                b = p['endemic_coef'][:, sin_i, hh]
                c = p['endemic_coef'][:, cos_i, hh]
                amp   = np.sqrt(b ** 2 + c ** 2)
                phase = np.arctan2(b, c)                       # peak of b*sin + c*cos
                # week feature: theta = 2*pi*week/52, so the curve peaks at
                # week = phase * 52 / (2*pi) (feature time t0); shift to target time
                peak_t0 = phase / (2 * np.pi) * 52
                lead  = self.epiconfig.horizon_leadtime + hh
                df['seasonal_amplitude'] = np.exp(2 * amp)
                df['endemic_peak_week']  = ((peak_t0 + lead - 1) % 52) + 1     # in [1, 53)
            if 'epidemic_rate' in p:
                df['epidemic_rate'] = p['epidemic_rate'][:, hh]
            if 'neighbourhood_rate' in p:
                df['neighbourhood_rate'] = p['neighbourhood_rate'][:, hh]
            df['epidemic_multiplier']      = p['epidemic_multiplier']
            df['neighbourhood_multiplier'] = p['neighbourhood_multiplier']
            df['alpha'] = p['alpha']
            rows.append(df)

        out = pd.concat(rows, ignore_index=True)
        names = self.context_data.nodenames
        name_col = f'{self.epiconfig.level}_name'
        if id_col in names.columns and name_col in names.columns:
            out = out.merge(names[[id_col, name_col]].rename(columns={name_col: 'node_name'}),
                            on=id_col, how='left')
        return out
