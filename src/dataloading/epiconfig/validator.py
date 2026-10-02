from __future__ import annotations

from typing import assert_never, TYPE_CHECKING

from .exceptions import EpiConfigLimitationError, EpiConfigValidationError
from ...utils import ExceptionReport, PathNotFound

if TYPE_CHECKING:
    from .epiconfig import EpiConfig

import logging
logger = logging.getLogger(__name__)

class EpiConfigValidator:
    """
    This helper class of ``EpiConfig`` deals with the validation of input and the paths
    stored in ``EpiPathsManager``. Simply call `.validate()`. Warnings and Exceptions 
    will be returned as an ``ExceptionReport``; a nice formatted version of multiple 
    exceptions.

    Methods
    -------
    ``validate()``
        Validates everything.

    See Also
    --------
    ``ExceptionReport``
        An easy-to-read alternative of GroupedException, that also works well in jupyter
        notebooks.

    Downstream
    ----------
    ``EpiConfig`` stores all configuration information needed to transform raw data
    into model-ready datasets by ``EpiDataOrchestrator``. ``EpiValidator`` validates the
    input to ``EpiConfig``, as well as the paths that ``EpiPathsManager`` stores.    
    """

    def __init__(self,
                 epiconfig: EpiConfig):
        
        self.epiconfig = epiconfig 

    def validate(self):
        exceptions: list[Exception] = []
        exceptions = self._datapaths(exceptions)
        exceptions = self._current_limitations(exceptions)      
        exceptions = self._input(exceptions)  
        
        self._warnings()

        if len(exceptions) > 0:
            raise ExceptionReport(exceptions, context = "EpiConfig could not be created")        

        logger.debug('EpiConfig has been validated')            

    def _datapaths(self, exceptions: list[Exception]) -> list[Exception]:

        for property in self.epiconfig.path_manager.properties:

            path_attr = self.epiconfig.path_manager.get(property)

            if not path_attr.exists():
                exceptions.append(PathNotFound(path_attr)) 

        return exceptions        
    
    def _current_limitations(self, exceptions: list[Exception]) -> list[Exception]:
        """
        Validates any issues in the initialization of an EpiConfig instance. 
        These represent CURRENT limitations, which are also things for me to develop further.
        An CurrentEpiConfigError is thrown suggesting to adjust the input.
        """

        # temporal frequency
        if self.epiconfig.temporal_frequency not in ['m','w']:
            exceptions.append(EpiConfigLimitationError(f'invalid valid for temporal_frequency (currently). Value must be in ["m","w"]'))         

        return exceptions
    
    def _input(self, exceptions: list[Exception]) -> list[Exception]:
        """
        Validates discrepancies in the initialization of an EpiConfig instance. These represent
        actual issues or errors, so an EpiConfigError is thrown suggesting to adjust the input.
        """
        # temporal-related 
        if self.epiconfig.horizon_size < 1:
            exceptions.append(EpiConfigValidationError(f"horizon_size must be >= 1, got {self.epiconfig.horizon_size}"))
        
        if self.epiconfig.horizon_leadtime < 1:
            exceptions.append(EpiConfigValidationError(f"horizon_leadtime must be >= 1, got {self.epiconfig.horizon_leadtime}"))
        
        if self.epiconfig.sequence_length < 1:
            exceptions.append(EpiConfigValidationError(f"sequence_length must be >= 1, got {self.epiconfig.sequence_length}"))
        
        if self.epiconfig.lag_num < 1:
            exceptions.append(EpiConfigValidationError(f"number of lags must be >= 1, got {self.epiconfig.lag_num}"))


        # country-related
        match (self.epiconfig.country, self.epiconfig.level):

            case ('germany', 'nuts1' | 'nuts2' | 'nuts3'):
                pass

            case ('hungary', 'nuts1' | 'nuts2' | 'nuts3'):
                pass            

            case _:
                assert_never((self.epiconfig.country, self.epiconfig.level))                
        return exceptions
        
    def _warnings(self):
        """
        Validates some combinations of inputs that are likely not meant as such, and shouldn't disrupt the pipeline any further. 
        A EpiConfigWarning is thrown, not an exception
        """
        match (self.epiconfig.country, self.epiconfig.level):

            case ('hungary', 'nuts1'):
                logger.warning('Hungary - nuts1 units are very large (n = 3). Predictions may not be particulary informative.')

            case ('hungary', 'nuts2'):
                logger.warning('Hungary - nuts2 units are very large (n = 8). Predictions may not be particulary informative.')             

            case ('germany', 'nuts1'):
                logger.warning('Germany - nuts1 units are very large (n = 16). Predictions may not be particulary informative.')   
    
    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}>"
        return representation 