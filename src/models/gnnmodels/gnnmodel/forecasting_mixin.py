from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, assert_never
import pandas as pd
import torch
from torch import Tensor as Tensor
import numpy as np

from ....utils.types import DataSetSplit
from ..utils import UnexpectedDataShape
from ...utils import ModelStatus
from ...utils.conformal import residual_quantile_table
from ....dataloading.databuilders import GraphDataBuilder

if TYPE_CHECKING:
    from ....dataloading import ColumnRegistry
    from ...utils import PredictionManager
    from ....dataloading import EpiConfig
    from ..utils import Strategy, LossManager
    from ....dataloading.epidataorchestration.containers import ContextEpiData

# raw network output, stacked over timesteps:
#   point    -> Tensor [T, N, H]
#   quantile -> Tensor [T, N, H, Q]
#   nb       -> (mu [T, N, H], alpha [T, N, H])
RawOutput = Tensor | tuple[Tensor, Tensor]


class GNNModelForecastMixin:
    """
    Mixin class to ``GNNModel`` that deals with forecasting of models.

    Output heads and how they become prediction columns
    ---------------------------------------------------
    ============  ==========  ==================================================
    output_head   mode        prediction columns
    ============  ==========  ==================================================
    point         point       ``pred`` = network output
    point         interval    ``pred_q*`` = point + split-conformal offsets
                              (val residuals per horizon and week of year)
    quantile      interval    ``pred_q*`` = network quantiles, optionally CQR
    nb            point       ``pred`` = mu
    nb            interval    ``pred_q*`` = exact NB2(mu, alpha) quantiles
    ============  ==========  ==================================================

    Conformal and CQR corrections are computed on the model's working scale
    (the transformed target for point/quantile heads) and on the val split only.
    Val is also used for early stopping, so the calibration is slightly
    optimistic; a separate calibration split would remove this.
    """
    model:              torch.nn.Module
    databuilder:  GraphDataBuilder
    strategy:           Strategy
    verbose:            int
    epiconfig:          EpiConfig
    device:             torch.device
    loss:               LossManager
    predictions:        PredictionManager
    context_data:       ContextEpiData
    column_registration: ColumnRegistry
    output_head:        str
    conformalize:       bool
    calibration_offsets: dict[int, Any]

    # conformal wrapper settings (point head in interval mode)
    conformal_min_bin_obs: int = 30

    def forecast(self, dataset: DataSetSplit = 'test'):
        """forecast the given dataset"""
        self._check_status(['model_hparams_set', 'global_hparams_set', 'trained'])

        raw, targets, avg_loss = self._run_inference(dataset)
        setattr(self, f'{dataset}_loss', avg_loss)

        num_timesteps, num_nodes, horizon_size = targets.shape
        pred_cols = self.column_registration.pred_columns

        preds = self._raw_to_prediction_array(raw)                  # [T, N, H, C]

        if self.epiconfig._prediction_mode == 'interval':
            if self.output_head == 'point':
                preds = self._apply_conformal(preds[..., 0], dataset)
            elif self.output_head == 'quantile' and self.conformalize:
                preds = self._apply_cqr(preds, dataset)

        if preds.shape[-1] != len(pred_cols):
            raise UnexpectedDataShape(f'{preds.shape[-1]} prediction columns',
                                      f'{len(pred_cols)} ({pred_cols})',
                                      'formatting forecasts')

        results = self._format_forecast_results(
            predictions     = preds,
            targets         = targets.numpy(),
            dataset         = dataset,
            pred_col_names  = pred_cols,
        )

        for hh in range(horizon_size):
            horizon_cols = (
                [self.epiconfig.temporal_column, self.epiconfig.id_column]
                + [f'{c}_{hh}' for c in pred_cols] + [f'target_{hh}']
            )
            rename = {f'{c}_{hh}': c for c in pred_cols}
            rename[f'target_{hh}'] = 'target'

            horizon_data = results[horizon_cols].rename(columns=rename)
            self.predictions.add_horizon_predictions(dataset, horizon_data, hh)

        self._update_status('forecasted')

    # ======================================================================= #
    # inference
    # ======================================================================= #
    def _get_dataloader(self, dataset: DataSetSplit):
        match dataset:
            case 'train':
                return self.databuilder.dataloader_train
            case 'val':
                return self.databuilder.dataloader_val
            case 'test':
                return self.databuilder.dataloader_test
            case _:
                assert_never(dataset)

    def _run_inference(self, dataset: DataSetSplit) -> tuple[RawOutput, Tensor, float]:
        """
        Run the network over a split. Returns the stacked raw output, the targets
        [T, N, H] and the average loss.
        """
        self.model.eval()
        dataloader = self._get_dataloader(dataset)

        num_nodes = self.context_data.num_nodes
        H         = self.epiconfig.horizon_size

        outs:    list[RawOutput] = []
        targets: list[Tensor]    = []
        total_loss = 0.0

        with torch.no_grad():
            for idx, snapshot in enumerate(dataloader):
                snapshot = snapshot.to(self.device)

                y_hat, loss_val = self.strategy.forecast_step(
                    model   = self.model,
                    snapshot= snapshot,
                    loss_fn = self.loss
                )
                total_loss += loss_val

                if idx == 0:
                    self._validate_output_shape(y_hat, num_nodes, H)

                if isinstance(y_hat, tuple):
                    outs.append(tuple(t.detach().cpu() for t in y_hat))
                else:
                    outs.append(y_hat.detach().cpu())
                targets.append(snapshot.y.detach().cpu())

        targets_tensor = torch.stack(targets)
        expected = [len(dataloader), num_nodes, H]
        if list(targets_tensor.shape) != expected:
            raise UnexpectedDataShape(f'{list(targets_tensor.shape)}', f'{expected}', 'stacked raw targets')

        if isinstance(outs[0], tuple):
            raw: RawOutput = (torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs]))
        else:
            raw = torch.stack(outs)   # type: ignore[arg-type]

        return raw, targets_tensor, total_loss / len(dataloader)

    def _validate_output_shape(self, y_hat: RawOutput, num_nodes: int, H: int) -> None:
        if self.output_head == 'nb':
            if not isinstance(y_hat, tuple) or len(y_hat) != 2:
                raise UnexpectedDataShape(f'{type(y_hat)}', '(mu, alpha) tuple', 'nb forecast output')
            for t in y_hat:
                if list(t.shape) != [num_nodes, H]:
                    raise UnexpectedDataShape(f'{list(t.shape)}', f'{[num_nodes, H]}', 'nb mu/alpha')
            return

        assert isinstance(y_hat, Tensor)
        if self.output_head == 'quantile':
            expected = [num_nodes, H, self.epiconfig._num_quantiles]
        else:
            expected = [num_nodes, H]
        if list(y_hat.shape) != expected:
            raise UnexpectedDataShape(f'{list(y_hat.shape)}', f'{expected}', 'forecast output snapshot 0')

    def _raw_to_prediction_array(self, raw: RawOutput) -> np.ndarray:
        """Convert raw output to [T, N, H, C] with C = 1 (point) or Q (quantiles)."""
        if self.output_head == 'nb':
            assert isinstance(raw, tuple)
            mu, alpha = raw[0].numpy(), raw[1].numpy() * getattr(self, 'alpha_scale', 1.0)
            if self.epiconfig._prediction_mode == 'interval':
                from ..architectures.modules.hhh4module import nb_quantiles
                return nb_quantiles(mu, alpha, self.epiconfig.quantiles)
            return mu[..., None]

        assert isinstance(raw, Tensor)
        arr = raw.numpy()
        return arr if arr.ndim == 4 else arr[..., None]

    # ======================================================================= #
    # calibration on val
    # ======================================================================= #
    def _target_week_index(self, dataset: DataSetSplit, num_timesteps: int, hh: int) -> np.ndarray:
        """Seasonal index of the TARGET time (t0 + leadtime + hh) per timestep."""
        t0 = pd.to_datetime(pd.Series(self._t0_timestamps(dataset, num_timesteps)))
        steps = self.epiconfig.horizon_leadtime + hh

        if self.epiconfig.temporal_frequency == 'w':
            return (t0 + pd.Timedelta(weeks=steps)).dt.isocalendar().week.astype(int).to_numpy()
        if self.epiconfig.temporal_frequency == 'm':
            return (t0 + pd.DateOffset(months=steps)).dt.month.astype(int).to_numpy()
        raise ValueError(f'unsupported temporal frequency {self.epiconfig.temporal_frequency}')

    def _apply_conformal(self, point: np.ndarray, dataset: DataSetSplit) -> np.ndarray:
        """
        Split-conformal intervals around a point forecast [T, N, H].

        Residual quantiles from val, per horizon and week of year, with the same
        thin-bin fallback as the baselines. Returns [T, N, H, Q].
        """
        quantiles = self.epiconfig.quantiles
        assert quantiles is not None

        val_raw, val_y, _ = self._run_inference('val')
        val_point = self._raw_to_prediction_array(val_raw)[..., 0]
        Tv, N, H  = val_point.shape
        T         = point.shape[0]

        out = np.empty(point.shape + (len(quantiles),), dtype=float)

        for hh in range(H):
            val_week = np.repeat(self._target_week_index('val', Tv, hh), N)
            table = residual_quantile_table(
                target    = pd.Series(val_y[:, :, hh].numpy().ravel()),
                pred      = pd.Series(val_point[:, :, hh].ravel()),
                group     = pd.Series(val_week),
                quantiles = quantiles,
                scale     = 'additive',
                min_obs   = self.conformal_min_bin_obs,
            )
            self.calibration_offsets[hh] = table

            week = pd.Series(np.repeat(self._target_week_index(dataset, T, hh), N))
            q_df = table.apply(pd.Series(point[:, :, hh].ravel()), week, clip_lower=None)
            out[:, :, hh, :] = q_df[quantiles].to_numpy().reshape(T, N, len(quantiles))

        return out

    def _apply_cqr(self, preds: np.ndarray, dataset: DataSetSplit) -> np.ndarray:
        """
        Conformalized quantile regression (Romano et al., 2019) per horizon and
        per central interval. For the pair (q, 1-q), the conformity score is
        ``max(lo - y, y - hi)`` on val; both bounds move by its
        ceil((n+1)(1-2q))/n empirical quantile. The median is untouched.
        """
        quantiles = self.epiconfig.quantiles
        assert quantiles is not None
        Q, mid = len(quantiles), len(quantiles) // 2

        val_raw, val_y, _ = self._run_inference('val')
        val_q  = self._raw_to_prediction_array(val_raw)                 # [Tv, N, H, Q]
        y      = val_y.numpy()

        out = preds.copy()
        corrections: dict[float, np.ndarray] = {}

        for i in range(mid):
            lo, hi = i, Q - 1 - i
            level  = 1.0 - 2.0 * quantiles[i]
            scores = np.maximum(val_q[..., lo] - y, y - val_q[..., hi])  # [Tv, N, H]
            scores = scores.reshape(-1, scores.shape[-1])                # [Tv*N, H]
            n      = scores.shape[0]
            k      = min(1.0, math.ceil((n + 1) * level) / n)
            corr   = np.quantile(scores, k, axis=0, method='higher')     # [H]

            out[..., lo] -= corr
            out[..., hi] += corr
            corrections[quantiles[i]] = corr

        # inner pairs can be widened more than outer ones; restore the order
        out = np.sort(out, axis=-1)
        self.calibration_offsets = {'cqr': corrections}
        return out

    # ======================================================================= #
    # formatting
    # ======================================================================= #
    def _t0_timestamps(self, dataset: DataSetSplit, num_timesteps: int) -> np.ndarray:
        """t0 timestamp (last observed step) of each snapshot in a split."""
        global_indices = self.databuilder.time_splits[
            self.databuilder.time_splits[dataset]
        ].index

        # train snapshots start once a full window is available
        offset = (self.databuilder.dataorchestrator.config.sequence_length - 1) if dataset == 'train' else 0

        return self.databuilder.time_splits.loc[
            global_indices[np.arange(num_timesteps) + offset], self.epiconfig.temporal_column
        ].values

    def _format_forecast_results(
        self,
        predictions:    np.ndarray,
        targets:        np.ndarray,
        dataset:        Literal['train','val','test'],
        pred_col_names: list[str],
        ) -> pd.DataFrame:
        """
        Formats predictions into a flat DataFrame aligned with correct timestamps.

        predictions shape: [num_timesteps, num_nodes, horizon_size, num_pred_cols]
        targets shape:     [num_timesteps, num_nodes, horizon_size]
        """
        num_timesteps, num_nodes, horizon_size, _ = predictions.shape

        sequence_idx = np.repeat(np.arange(num_timesteps), num_nodes)
        node_idx     = np.tile(np.arange(num_nodes), num_timesteps)

        t0         = self._t0_timestamps(dataset, num_timesteps)
        timestamps = t0[sequence_idx]

        results = pd.DataFrame({
            self.epiconfig.temporal_column: timestamps,
            self.epiconfig.id_column: node_idx,
        })

        # Sanity check: first and last timestamp should match expected range
        expected = self.predictions.temporal_summary.get_daterange_dataset(dataset, reference='t0')
        assert pd.Timestamp(timestamps[0]) == pd.Timestamp(expected[0]), \
            f"First timestamp mismatch: got {timestamps[0]}, expected {expected[0]}"
        assert pd.Timestamp(timestamps[-num_nodes]) == pd.Timestamp(expected[1]), \
            f"Last timestamp mismatch: got {timestamps[-num_nodes]}, expected {expected[1]}"

        pred_flat   = predictions.reshape(num_timesteps * num_nodes, horizon_size, -1)
        target_flat = targets.reshape(num_timesteps * num_nodes, horizon_size)

        for hh in range(horizon_size):
            for cc, col_name in enumerate(pred_col_names):
                results[f'{col_name}_{hh}'] = pred_flat[:, hh, cc]
            results[f'target_{hh}'] = target_flat[:, hh]

        return results

    # ========== STUBS ========== #
    def _check_status(self, required_states: list[ModelStatus] | ModelStatus) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...
