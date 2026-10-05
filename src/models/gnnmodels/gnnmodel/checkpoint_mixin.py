from __future__ import annotations

from typing import Any, TYPE_CHECKING
import torch 
from torch import Tensor as Tensor
from pathlib import Path

from ...utils import ModelInitError
from ....utils import PathNotFound

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig

class GNNModelCheckpointMixin:
    """ 
    Mixin class to ``GNNModel`` that deals with saving of models.
    """    
    model: torch.nn.Module 
    clean_name: str
    models_dir: Path
    config_info: dict[str, Any]
    verbose: int
    epiconfig: EpiConfig

    def save_model(self, dir: Path) -> None:
        """
        Save model

        Parameters
        ----------
        dir : Path
            Directory in which to save the model. NOTE that this directory
            needs to exist.
        """
        if not hasattr(self, 'model'):
            raise ModelInitError('No attribute "model" found')
        
        # if parent of dir doesn't exist, error
        if not dir.parent.exists():
            raise PathNotFound(f"parent of dir {dir} does not exist")

        filepath = dir / f"{self.clean_name}.pt"

        save_dict: dict[str, Any] = {
            'name':               self.clean_name,            
            'model_class':        self.__class__.__name__,
            'model_state':        self.model.state_dict(),
            'model_hparams':      self.config_info.get('model_hparams', {}),
            'global_hparams':     self.config_info.get('global_hparams', {}),
            'epiconfig_summary':  self.epiconfig.get_summary(level = 1),
            'monitoring_metrics': getattr(self, 'monitoring_metrics', None),
        }    

        torch.save(save_dict, filepath)