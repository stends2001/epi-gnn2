from torch import Tensor as Tensor

from .baseloss import BaseLoss
from ..exceptions import InvalidLossError

class LossManager:
    """
    Manages loss functions. Handles instantiation and the usage. Loss functions are 
    called through ``__call__()``.

    Parameters
    ----------
    loss_name : str
        The name of hte loss function used. If not available in the registry, exception
        is raised.

    See Also
    --------
    ``BaseLoss``
        Parent class to all Loss functions.
    """

    def __init__(self, 
                 loss_name: str):
        if loss_name not in BaseLoss._registry:
            available = list(BaseLoss._registry.keys())
            raise InvalidLossError(
                f'{loss_name}', available
            )
        
        self.loss_name  = loss_name
        self.loss_fn    = BaseLoss._registry[loss_name]()
    
    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:     
        return self.loss_fn(y_hat, y)
    
    def __repr__(self) -> str:
        return f"LossHandler({self.loss_fn})"
    
    @staticmethod
    def list_available_losses() -> list[str]:
        """Return list of available loss functions."""
        return list(BaseLoss._registry.keys())
