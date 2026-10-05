from typing import Self, Union, Type, Any
import torch 
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path 
import pandas as pd

from .internals_mixin import GNNModelInternalsMixin
from .presentation_mixin import GNNModelPresentationMixin
from .training_mixin import GNNModelTrainMixin
from .forecasting_mixin import GNNModelForecastMixin
from .globalhparams_mixin import GNNModelGlobalhParamsMixin
from .checkpoint_mixin import GNNModelCheckpointMixin

from ....dataloading.databuilders import GraphDataBuilder
from ..utils import Strategy
from ...basemodel import BaseModel

class GNNModel(
    GNNModelInternalsMixin,
    GNNModelPresentationMixin,
    GNNModelTrainMixin,
    GNNModelForecastMixin,
    GNNModelGlobalhParamsMixin,
    GNNModelCheckpointMixin,
    BaseModel # BaseModel comes last for hierarchy of methods-imported
    ):
    """ 
    Parent class to all GNN model architectures (``GCNModel`` and ``GATModel``).
    This is a first - order subclass to ``BaseModel``.

    Parameters
    ----------
    databuilder : GraphDataBuilder
        Data builder for the model to train on and predict from.
    strategy : Strategy
        Utility class for the training/validation/testing steps.
    name : str
        Name of the model.

    See Also
    --------
    ``BaseModel``
        Parent class to all model classes.

    ``Strategy``
        Utility class for the training/validation/testing steps.

    ``LossManager``
        Utility class that deals with the definition and usage of loss.

    ``GNNModelInternalsMixin``
        Mixin class that deals with some internal attributes-setting.
    ``GNNModelPresentationMixin``
        Mixin class that deals with representation and print-output.
    ``GNNModelTrainMixin``
        Mixin class that deals with the training of models.
    ``GNNModelForecastMixin``
        Mixin class that deals with the forecasting of models.
    ``GNNModelGlobalhParamsMixin``
        Mixin class that deals with setting and validating the global hyper parameters.
    ``GNNModelCheckpointMixin``
        Mixin class that deals with saving of models.
    
    Downstream
    ----------
    Model classes are sub classes to ``BaseModel``. The first order of sub classes are
    ``BaseLineModel`` and ``GNNModel``. These in turn, have model-specific subclasses.

    ``GNNModel`` subclasses are ``GATModel`` and ``GCNModel``. Note that the Model
    architectures are defined in ``../architectures/modules`` and the way that these
    are used are defined in models, in ``../architectures/modularchitectures``.
    """
    
    _childclasses:  dict[str, Type[Self]] = {}    
    model:          torch.nn.Module 
    optimizer:      Optimizer
    scheduler:      _LRScheduler

    def __init__(self, 
                 databuilder: GraphDataBuilder, 
                 strategy: Strategy,
                 name: str):

        # DeepModel in itself may not be initted
        if self.__class__ is GNNModel:
            raise TypeError("DeepModel cannot be instantiated directly")

        super().__init__(databuilder, name)        
    
        self.evaluation_datasets                            = {}
        self._residual_quantiles: dict[tuple[int, int], dict[int, float]] = {}

        # using hidden methods in DeepModelInternalsMixin, set attributes
        self._set_device()
        self._set_strategy(strategy)
        self._set_models_directory()

    # ======= DUNDER ======= #
    # __init_subclass__ is run when a subclass is iniated
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        GNNModel._childclasses[cls.__name__.lower()] = cls

    # ======= MAIN METHODS =========== #
    @classmethod
    def load_model(cls,
                   model_name:          str,
                   databuilder:   GraphDataBuilder,
                   dir:                 Union[str, Path],
                   ) -> Self:
        """
        Loads a saved model, and sets model hyper-parameters, global-hyperparameters
        and thus, most importantly, self.model. This function does not, however, 
        run `forecast()`! So this needs to be done after loading.

        Parameters
        ----------
        model_name: str
            name under which the model is saved (should be the filename wihtout .pt)
        dataloadermanager: Union[GraphDataLoaderManager, DeepDataLoaderManager]
            dataloadermanager with which the model was trained.
        subdir: Optional[str] = None
            directory in which to find the model. Directory may be named after an experiment.

        Returns
        -------
        This is the only method that returns the instance of the model.
        """

        if isinstance(dir, str):
            dir = Path(dir)
        
        # construct model path
        if model_name.endswith('.pt'):
            filepath = dir / model_name
        else:
            filepath = dir / f"{model_name}.pt"

        # validate model path's existence
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        # load dictionary
        save_dict: dict[str, Any] = torch.load(filepath, map_location='cpu', weights_only=False)

        # get name of model architecture - class
        model_key: str = save_dict['model_class'].lower()

        # if class doesn't exist in deepmodel's child classes, then raise error
        if model_key not in cls._childclasses:
            raise ValueError(
                f"Unknown model class '{save_dict['model_class']}'. "
                f"Available: {list(cls._childclasses.keys())}"
            )

        # create an instance of model
        child_cls = cls._childclasses[model_key]
        instance  = child_cls(
            name              = save_dict['name'],
            databuilder = GraphDataBuilder,
        ) # type: ignore
        
        databuilder.dataorchestrator.config.assert_equals(save_dict['epiconfig_summary'], level = 1)

        # compare between raw input timestamps, not the preprocessed ones in temporal_summary
        saved_test_start  = pd.Timestamp(save_dict['epiconfig_summary']['split_valtest'])
        new_train_end     = pd.Timestamp(databuilder.dataorchestrator.config.split_valtest)

        if new_train_end > saved_test_start:
            raise ValueError(
                f"Data leakage: new dataloader's train/val period ({new_train_end}) "
                f"overlaps with saved model's test period ({saved_test_start})"
            )

        # load config into the model
        instance.set_model_hparams(**save_dict['model_hparams'])
        instance.set_global_hparams(**save_dict['global_hparams'])
        instance.model.load_state_dict(save_dict['model_state'])
        instance.model.to(instance.device)
        instance.monitoring_metrics           = save_dict['monitoring_metrics']
        instance.config_info['model_hparams'] = save_dict['model_hparams']
        instance.config_info['global_hparams']= save_dict['global_hparams']
        instance._update_status('trained')

        return instance

    def set_model_hparams(self) -> None:
        """must be set by subclasses"""
        raise NotImplementedError("Subclass of DeepModel must implement set_model_hparams")      