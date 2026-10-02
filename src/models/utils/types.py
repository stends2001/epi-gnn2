from typing import Union, Literal 

SingleNodeType  = Union[int,Literal['national']]
ModelStatus     = Literal['model_initialized', 'model_hparams_set', 'global_hparams_set','trained','forecasted']