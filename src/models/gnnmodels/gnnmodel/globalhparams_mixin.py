from __future__ import annotations

from typing import Any, TYPE_CHECKING
import torch 
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..utils import InvalidOptimizerError, InvalidSchedulerError
from ..utils import LossManager
from ...utils import ModelStatus

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig

class GNNModelGlobalhParamsMixin:
    """ 
    Mixin class to ``GNNModel`` that deals with setting and validating of global
    hyperparameters.    
    """    
    status_dict:        dict[ModelStatus, bool]
    epiconfig:          EpiConfig
    config_info:        dict[str, Any]    
    model:              torch.nn.Module
    n_params:           int

    def set_global_hparams(self, 
                           lr:              float           = 0.001,
                           n_epochs:        int             = 5,
                           patience:        int             = 15,
                           min_delta:       float           = 1e-4,                            
                           optimizer:       str             = 'adam',
                           loss:            str             = 'mse',                           
                           scheduler:       str             = 'step',
                           # kwargs
                           optimizer_kwargs: dict[str, Any] | None = None,                           
                           scheduler_kwargs: dict[str, Any] | None = None,                    
                           ) -> None:
        """
        Prepares model for training by setting global hyperparameters.
        
        Parameters
        ---------
        lr : float = 0.001
            Learning rate.
        n_epochs : int = 5
            Number of epochs to train the model.
        patience : int = 15
            Number of epochs without improvement before interrupting training.
        min_delta: float = 1e-4                     
            Minimal change in loss to consider 'improvement'.
        optimizer: str = 'adam'
            Optimizer to use when training. Options can be found in `_get_optimizer()`.
        loss: str = 'mse'                          
            Loss to use when training. Options can be found in `LossHandler`.
        Scheduler: str = 'step'
            Scheduler to use when training. Options can be found in `_get_scheduler()`.
        optimizer_kwargs:Optional[Dict[str, Any]] = None  
            any kwargs relevant to optimizer                         
        scheduler_kwargs:Optional[Dict[str, Any]] = None
            any kwargs relevant to scheduler
        """
        self._check_status(['model_hparams_set'])

        global_hparams_config: dict[str, Any] = {
            'lr'                : lr,
            'n_epochs'          : n_epochs,
            'patience'          : patience,
            'min_delta'         : min_delta,                       
            'optimizer'         : optimizer,
            'loss'              : loss,
            'scheduler'         : scheduler,

            'optimizer_kwargs'  : optimizer_kwargs,
            'scheduler_kwargs'  : scheduler_kwargs
        }
        
        # ==== CONSTANTS ===== #
        self.n_epochs           = n_epochs
        self.patience           = patience
        self.min_delta          = min_delta

        # ==== LOSS ==== #
        self.loss       = LossManager(loss)  

        # ==== OPTIMIZER ==== #
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = self._get_optimizer(optimizer, lr, optimizer_kwargs)
        
        # ==== SCHEDULER ====== #
        if scheduler_kwargs is None:
            default_scheduler_kwargs : dict[str, Any] = {
                'step':        {'step_size': 15, 'gamma': 0.8},
                'exponential': {'gamma': 0.95},
                'cosine':      {'T_max': 50},
                'cosine_warm': {'T_0': 10, 'T_mult': 2},                
                'plateau':     {'mode': 'min', 'factor': 0.5, 'patience': 10, 'verbose': True}
            }
            scheduler_kwargs = default_scheduler_kwargs.get(scheduler, {}) if scheduler else {}

        self.scheduler = self._get_scheduler(scheduler, self.optimizer, scheduler_kwargs)
        self.config_info['global_hparams']  = global_hparams_config
        self._update_status('global_hparams_set')


    def _get_optimizer(self, 
                       optimizer_name:  str, 
                       lr:              float, 
                       optimizer_kwargs: dict[str, Any]) -> Optimizer:
        """Factory method to create and return optimizer"""
        
        self._check_status(['model_hparams_set'])   

        # pylance struggles with torch typing?
        optimizer_map : dict[str, Any] = {
            'adam':    optim.Adam,     # type: ignore
            'adamw':   optim.AdamW,    # type: ignore
            'sgd':     optim.SGD,      # type: ignore
            'rmsprop': optim.RMSprop,  # type: ignore
            'adagrad': optim.Adagrad,  # type: ignore
        }
        
        if optimizer_name.lower() not in optimizer_map:
            raise InvalidOptimizerError(optimizer_name, list(optimizer_map.keys()))
        
        optimizer_class = optimizer_map[optimizer_name.lower()]

        return optimizer_class(self.model.parameters(), lr=lr, **optimizer_kwargs)

    def _get_scheduler(self, scheduler_name: str, optimizer: Optimizer, scheduler_kwargs: dict[str, Any]) -> _LRScheduler:
        """Factory method to create and return scheduler"""
        
        self._check_status(['model_hparams_set'])
        
        scheduler_map : dict[str, Any] = {
            'step':        torch.optim.lr_scheduler.StepLR,
            'exponential': torch.optim.lr_scheduler.ExponentialLR,
            'cosine':      torch.optim.lr_scheduler.CosineAnnealingLR,
            'cosine_warm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,            
            'plateau':     torch.optim.lr_scheduler.ReduceLROnPlateau,
            'cyclic':      torch.optim.lr_scheduler.CyclicLR,
            'onecycle':    torch.optim.lr_scheduler.OneCycleLR,
            'multistep':   torch.optim.lr_scheduler.MultiStepLR,
            'lambda':      torch.optim.lr_scheduler.LambdaLR,
        }
        
        if scheduler_name.lower() not in scheduler_map:
            raise InvalidSchedulerError(scheduler_name, list(scheduler_map.keys()))
        
        scheduler_class = scheduler_map[scheduler_name.lower()]
        return scheduler_class(optimizer, **scheduler_kwargs)
    
    # ======== STUBS ======= #
    def _check_status(self, required_states: list[ModelStatus] | ModelStatus) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...