from torch.optim.optimizer import Optimizer
import torch

from .lossmanager import LossManager
from ....dataloading.databuilders.graphdatabuilder.datacontainers import Data

Output = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Strategy:
    """
    Handles training and forecasting.

    The model output is passed to the loss unchanged, so it can be a tensor
    ([N, H] point or [N, H, Q] quantiles) or a tuple ``(mu, alpha)`` for the NB
    head; the matching loss interprets it.
    """
    def _detach_and_move(self, 
                         state : torch.Tensor | None, 
                         device : torch.device) -> torch.Tensor | None:
        """move tensor to device"""
        if state is None:
            return None
        
        return state.detach().to(device)    

    def training_step(self, 
                      model : torch.nn.Module, 
                      snapshot : Data, 
                      optimizer : Optimizer, 
                      loss_fn : LossManager) -> float:
        """
        Single training step that returns the loss.

        steps:
        
        - reset gradients
        - make predictions
        - get loss
        - compute new gradient
        - perform single optimization step
        """
        y_hat:  Output
        loss:   torch.Tensor        
        
        optimizer.zero_grad()

        assert snapshot.graph is not None

        y_hat = model(snapshot.x, 
                      snapshot.graph.edge_index, 
                      snapshot.graph.edge_weight)

        loss    = loss_fn(y_hat, snapshot.y)

        # optional model penalty (e.g. HHH4Module's ridge on node effects);
        # only added during training, so val/test losses stay comparable
        if hasattr(model, 'regularization'):
            loss = loss + model.regularization()

        loss.backward()
        optimizer.step()

        return loss.item()
    
    def validation_step(self, 
                        model : torch.nn.Module, 
                        snapshot : Data, 
                        loss_fn : LossManager) -> float:
        """
        Single validation step that returns the loss.

        steps:

        - make predictions
        - get loss
        """

        y_hat:  Output
        loss:   torch.Tensor      

        assert snapshot.graph is not None

        y_hat       = model(snapshot.x, snapshot.graph.edge_index, snapshot.graph.edge_weight)      

        loss = loss_fn(y_hat, snapshot.y)
        return loss.item()
    
    def forecast_step(self, 
                      model: torch.nn.Module, 
                      snapshot: Data, 
                      loss_fn: LossManager) -> tuple[Output, float]:
        """ 
        Single test step that returns the predictions and loss.

        steps:

        - make predictions
        - get loss
        """
        y_hat:  Output
        loss:   torch.Tensor     

        assert snapshot.graph is not None

        y_hat = model(snapshot.x, snapshot.graph.edge_index, snapshot.graph.edge_weight)

        loss = loss_fn(y_hat, snapshot.y)
        return y_hat, loss.item()


    def __repr__(self) -> str:
        return "standard strategy"