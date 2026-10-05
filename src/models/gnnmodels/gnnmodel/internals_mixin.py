import torch 

from ..utils import Strategy
from ....utils import PathManager

class GNNModelInternalsMixin:
    """
    Mixin class to ``GNNModel`` that deals with some internal attributes-setting.     
    """
    def _set_strategy(self, strategy: Strategy):
        """sets strategy"""
        self.strategy = strategy

    def _set_device(self):
        """sets device"""
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        if self.device.type == 'cpu':
            print('device found is CPU')

    def _set_models_directory(self):
        """sets model directory"""
        self.models_dir = PathManager().outcomes  
