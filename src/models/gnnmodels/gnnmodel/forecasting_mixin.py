from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never
import pandas as pd
import torch 
from torch import Tensor as Tensor
import numpy as np

from ....utils.types import DataSetSplit
from ..utils import UnexpectedDataShape
from ...utils import ModelStatus
from ....dataloading.databuilders import GraphDataBuilder

if TYPE_CHECKING:
    from ....dataloading import ColumnRegistry
    from ...utils import PredictionManager
    from ....dataloading import EpiConfig
    from ..utils import Strategy, LossManager
    from ....dataloading.epidataorchestration.containers import ContextEpiData    

class GNNModelForecastMixin:
    """
    Mixin class to ``GNNModel`` that deals with forecasting of models.    
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
    _residual_quantiles: dict[tuple[int, int], dict[int, float]]  

    def forecast(self, dataset: DataSetSplit = 'test'):
        """forecast the given dataset"""
        raw_predictions: list[Tensor]   = []
        raw_targets: list[Tensor]       = []

        # check the required states
        self._check_status(['model_hparams_set', 'global_hparams_set', 'trained'])

        # set model in evaluation mode
        self.model.eval()

        match dataset:
            case 'train':
                dataloader = self.databuilder.dataloader_train
            case 'val':
                dataloader = self.databuilder.dataloader_val 
            case 'test':
                dataloader = self.databuilder.dataloader_test
            case _:
                assert_never(dataset)
        
        # define iterator: whether or not to use tqdm
        iterator            = dataloader
        total_loss          = 0
        
        # setup expected predictions-shape [num_nodes, horizon_size]
        num_nodes           = self.context_data.num_nodes
        expected_shape_yhat = [num_nodes, self.epiconfig.horizon_size]

        # turn off gradient tracking
        with torch.no_grad():
                
            # for each snapshot, forecast
            for idx, snapshot in enumerate(iterator):
                snapshot = snapshot.to(self.device)

                y_hat, loss_val = self.strategy.forecast_step(
                    model   = self.model, 
                    snapshot= snapshot, 
                    loss_fn = self.loss
                )

                total_loss += loss_val

                # validate predictions-shape only the first snapshot
                if idx == 0:
                    if list(y_hat.shape) != expected_shape_yhat:
                        raise UnexpectedDataShape(
                            f'{list(y_hat.shape)}', f'{expected_shape_yhat}', "stacked yhat forecasting snapshot 0"
                            )

                raw_predictions.append(y_hat.detach().cpu())
                raw_targets.append(snapshot.y.detach().cpu())
        
        avg_loss = total_loss / len(dataloader)
        setattr(self, f'{dataset}_loss', avg_loss)

        # =========== SHAPE CHECK 1 ============= #
        # At this point, raw_predictions is a List of len [timestamps].
        # at each idx, there is a Tensor with shape [num_nodes, horizon_size, quantiles].
        # Since quantiles don't play a role in the target, those do not have that final dim
        # We're now removing the list-ness and stack that to a new dimension. The tensors therefore
        # get 3 (target) and 4 (predictions) dimensions.

        predictions_tensor  = torch.stack(raw_predictions)
        targets_tensor      = torch.stack(raw_targets)

        expected_shape_predictions  = [len(dataloader), num_nodes, self.epiconfig.horizon_size]
        expected_shape_targets      = [len(dataloader), num_nodes, self.epiconfig.horizon_size]        

        received_shape_predictions  = list(predictions_tensor.shape)
        received_shape_targets      = list(targets_tensor.shape)

        if expected_shape_predictions != received_shape_predictions:
            raise UnexpectedDataShape(
                f'{received_shape_predictions}', f'{expected_shape_predictions}', "stacked raw predictions"
                )

        if expected_shape_targets != received_shape_targets:
            raise UnexpectedDataShape(
                f'{received_shape_targets}', f'{expected_shape_targets}', "stacked raw targets"
                )

        num_timesteps, num_nodes, horizon_size = predictions_tensor.shape

        pred_col = self.column_registration.pred_columns[0]
        
        results = self._format_forecast_results(
            predictions     = predictions_tensor,
            targets         = targets_tensor,
            dataset         = dataset,
            num_timesteps   = num_timesteps,
            num_nodes       = num_nodes,
            horizon_size    = horizon_size,
            pred_col_name   = pred_col,
        )

        for hh in range(horizon_size):
            # Select the columns for this horizon: timestamp, id, all pred_cols, target
            horizon_cols = (
                [self.epiconfig.temporal_column, self.epiconfig.id_column]
                + [f'{pred_col}_{hh}'] + [f'target_{hh}']
            )
            horizon_data = results[horizon_cols].rename(
                columns={f'{pred_col}_{hh}' : pred_col,
                          f'target_{hh}'    : 'target'}
            )

            self.predictions.add_horizon_predictions(dataset, horizon_data, hh)
        

        self._update_status('forecasted')

    def _format_forecast_results(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        dataset: Literal['train','val','test'],
        num_timesteps: int,
        num_nodes: int,
        horizon_size: int,
        pred_col_name: str,
        ) -> pd.DataFrame:
        """
        Formats predictions into a flat DataFrame aligned with correct timestamps.
        Handles both point forecasts (num_quantiles=1) and quantile forecasts.

        predictions shape: [num_timesteps, num_nodes, horizon_size]
        targets shape:     [num_timesteps, num_nodes, horizon_size]
        """

        # Reshape: [num_sequences * num_nodes, horizon_size]
        pred_reshaped = (
            predictions
            .view(num_timesteps * num_nodes, horizon_size)
            .numpy()
        )
        # Reshape: [num_sequences * num_nodes, horizon_size]
        target_reshaped = targets.view(num_timesteps * num_nodes, horizon_size).numpy()

        # Index arrays — np.repeat/tile is correct here, no issue
        sequence_idx = np.repeat(np.arange(num_timesteps), num_nodes)
        node_idx     = np.tile(np.arange(num_nodes), num_timesteps)

        global_indices = self.databuilder.time_splits[
            self.databuilder.time_splits[dataset]
        ].index

        offset = (self.databuilder.dataorchestrator.config.sequence_length - 1) if dataset == 'train' else 0

        timestamps = self.databuilder.time_splits.loc[
            global_indices[sequence_idx + offset], self.epiconfig.temporal_column
        ].values
        
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

        for hh in range(horizon_size):
            # for qq, col_name in enumerate(pred_col_names):
            results[f'{pred_col_name}_{hh}'] = pred_reshaped[:, hh]
            results[f'target_{hh}'] = target_reshaped[:, hh]

        return results

    # ========== STUBS ========== #
    def _check_status(self, required_states: list[ModelStatus] | ModelStatus) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...