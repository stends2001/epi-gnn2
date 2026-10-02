from typing import Any, Literal

from ..utils import PredictionManager, ModelInitError
from .forecastdisplay_mixin import ForecastDisplayMixin
from .appearance_mixin import ModelAppearanceMixin
from .status_mixin import ModelStatusMixin

from ...dataloading import BaseLineDataBuilder, GraphDataBuilder

class BaseModel(ModelStatusMixin, 
                ModelAppearanceMixin, 
                ForecastDisplayMixin):
    """ 
    Parent class of ALL models.
    Upon init, all models must supply the following

    Parameters
    ----------
    databuilder : BaseLineDataBuilder | GraphDataBuilder
        Data builder for the model: equivalent to a dataloader.        
    name: str
        Name of the model.

    Methods
    -------
    ``show_forecasts``
        Plot model forecasts.

    Attributes
    ----------
    ``epiconfig``
        Large configuration class that dictates which data to load.   
    ``column_registration``
        Registry of columns that keeps track of all columns and transformations.
    ``context_data``
        Data-orchestration container for context data.
    ``temporal_summary``
        Helper class that stores temporal information, built based on ``EpiConfig``.
    ``pred_col``
        Column name under which predictions are stored.

    ``model_color``
        Color of the model's predictions.
    ``clean_name``
        Cleaned version of the input name.
    
    See Also
    --------
    ``ModelStatusMixin``
        Mixin class that deals with the model's status.        
    ``ModelAppearanceMixin``
        Mixin class to that deals with the model's appearance 
        (colors, name, representation)
    ``ForecastDisplayMixin``
        Mixin class that deals with the visualizing the model's predictions.   

    ``PredictionsManager``
        Manages models' predictions.     

    Downstream
    -----------
    Model classes are sub classes to ``BaseModel``. The first order of sub classes are
    ``BaseLineModel`` and ``GNNModel``. These in turn, have model-specific subclasses.

    Examples
    --------
    ``BaseModel`` is not to be instantiated by itself.    
    """
    _expected_databuilder: Literal['BaseLineDataBuilder', 'GraphDataBuilder'] 
    config_info: dict[str, Any]

    def __init__(self, 
                 databuilder: BaseLineDataBuilder | GraphDataBuilder, 
                 name: str):

        # BaseModel in itself may not be initted
        if self.__class__ is BaseModel:
            raise ModelInitError("BaseModel cannot be instantiated directly")

        self.name = name
        self._set_databuilder_attributes(databuilder)

        # static attributes
        self.model_class = self.__class__.__name__
        self.model_color = self._get_model_color()
        self.clean_name = self._get_clean_name()
        
        # validate databuilder-type vs model-type
        # if unexpected -> Error is raised
        self._validate_databuilder()
        
        # dynamic (changing) attributes
        self.predictions = PredictionManager(self.databuilder.dataorchestrator, 
                                             self.column_registration, 
                                             self.temporal_summary)
        
        # Configuration - info
        self.config_info = {'name': self.name, 
                            'model_class': self.model_class}

        self._init_status()
        self._update_status('model_initialized')
        self._print_status_update('model_initialized')

    # ======== HIDDEN METHODS ========= #
    def _set_databuilder_attributes(self, databuilder: BaseLineDataBuilder | GraphDataBuilder):
        """An extention upon init: sets a range of easy to access attributes related to databuilder"""
        self.databuilder = databuilder
        self.epiconfig = self.databuilder.dataorchestrator.config
        self.column_registration = databuilder.dataorchestrator.column_registration
        self.context_data = databuilder.dataorchestrator.data_context
        self.temporal_summary = self.context_data.temporal_summary
        self.pred_col = self.epiconfig.pred_column

    def _validate_databuilder(self):
        """validate class of databuilder"""
        if not hasattr(self, '_expected_databuilder'):
            raise ModelInitError(f'attribute self._expected_databuilder not set in model {self.name}')

        expected = self._expected_databuilder
        got = self.databuilder.__class__.__name__

        if expected != got:
            raise ModelInitError(f'{self.name} expected a dataabuilder of class {expected} but got {got}')

    # ======== METHODS TO BE IMPLEMENTED BY SUBCLASSES ======== #
        # I'm using NotImplementedErrors over ABC-abstractmethods since 
        # some model-types need more arguments than other model-types.
    
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement train")

    def forecast(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement forecast")

    def set_global_hparams(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement set_global_hparams")

    def set_model_hparams(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement set_model_hparams")

    def save_model(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement save_model")
        