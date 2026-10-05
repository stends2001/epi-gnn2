from .baseloss import BaseLoss
import torch
import torch.nn as nn

class MSELoss(BaseLoss):
    """
    Standard Mean Squared Error loss.
    
    See Also
    --------
    ``BaseLoss``
        Parent class of all loss functions.    
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mse(y_pred, y_true)