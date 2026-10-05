from __future__ import annotations

import torch
import torch.nn as nn
from typing import Type, Self
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """
    Parent class for all Loss functions.

    Initially a wide range of loss functions were used, which is why this class exists.
    While this is not the case at this point, this provideds a nice clean way to extend
    loss functions later.

    Methods
    -------
    All loss classes inherit from ``torch.nn.Module`` and therefore employ the 
    ``forward`` function (called through ``LossManager.__call__()).

    See Also
    --------
    ``LossManager``
        Manages losses.
    """

    _registry: dict[str, Type[Self]] = {}
        
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    
    # init - subclass dunder. This code is reun at "class creation time", not at run-time.
    # so each class that inherits from this main BaseLoss runs this, and registers itself
    # as a loss method.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # strip 'loss' from classname and register
        # the class "MSELoss" is registered as "mse"
        key = cls.__name__.lower().replace('loss', '')
        BaseLoss._registry[key] = cls

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.compute(y_pred, y_true)

    @abstractmethod
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass
    
    def __repr__(self) -> str:
        kwargs_str = ', '.join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"
    
from .losses import MSELoss