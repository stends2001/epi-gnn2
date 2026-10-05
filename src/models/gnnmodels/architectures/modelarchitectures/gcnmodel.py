from ...utils import Strategy
from ...gnnmodel import GNNModel
from .....dataloading import GraphDataBuilder

from ..modules import GCNModule

class GCNModel(GNNModel):
    """
    """
    _expected_databuilder = 'GraphDataBuilder'
    def __init__(self,
                 databuilder: GraphDataBuilder,
                 name:              str           = 'gcnmodel'):

        super().__init__(
            databuilder   = databuilder,
            name                = name,
            strategy            = Strategy()
        )

    def set_model_hparams(self,
                          hidden_size:  int   = 64,
                          num_layers:   int   = 3,
                          dropout:      float = 0.2,
                          self_loops:   bool  = False,
                          norm_edges:   bool  = True,
                          residuals:    bool  = True):
        """
        """
        _num_features   = len(self.column_registration.get_entries_names_by_type('feature'))
        _num_nodes      = len(self.databuilder.dataorchestrator.data_context.local_shapedata)
        _horizon_size   = self.databuilder.dataorchestrator.config.horizon_size
        _seq_length     = self.databuilder.dataorchestrator.config.sequence_length

        self.model = GCNModule(
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            dropout_p       = dropout,
            self_loops      = self_loops,
            norm_edges      = norm_edges,
            residuals       = residuals,

            num_features    = _num_features,
            num_nodes       = _num_nodes,
            seq_length      = _seq_length,
            horizon_size    = _horizon_size
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'hidden_size':  hidden_size,
            'num_layers':   num_layers,
            'dropout':      dropout,
            'self_loops':   self_loops,
            'norm_edges':   norm_edges,
            'residuals' :   residuals
        }

        self._update_status('model_hparams_set')
