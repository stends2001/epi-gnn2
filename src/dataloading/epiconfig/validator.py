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
        exceptions = self._quantiles(exceptions)        
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
    
    def _quantiles(self, exceptions : list[Exception]) -> list[Exception]:
        
        if self.epiconfig._prediction_mode == 'point':
            
            if self.epiconfig.quantiles is not None:
                exceptions.append(EpiConfigValidationError(f'``_prediction_mode`` == "point". Excpected ``quantiles`` as ``None`` but got {self.epiconfig.quantiles}')) 

            if self.epiconfig._num_quantiles is not None:
                exceptions.append(EpiConfigValidationError(f'``_prediction_mode`` == "point". Excpected ``_num_quantiles`` as ``None`` but got {self.epiconfig._num_quantiles}')) 

        elif self.epiconfig._prediction_mode == 'interval':

            if self.epiconfig.quantiles is None:
                exceptions.append(EpiConfigValidationError(f'``_prediction_mode`` == "interval". Excpected ``quantiles`` as list of floats but got ``None``')) 

            else:
                for q in self.epiconfig.quantiles:

                    if q <= 0 or q >= 1:
                        exceptions.append(EpiConfigValidationError(f'Quantiles must be decimals: between 0 and 1. Got {q}')) 

                qs  = list(self.epiconfig.quantiles)
                mid = len(qs) // 2

                if len(qs) % 2 == 0 or abs(qs[mid] - 0.5) > 1e-9:
                    exceptions.append(EpiConfigValidationError(f'Number of quantiles must be an odd number centered around 0.5.'))

                # strictly increasing: the metrics, the monotone quantile head and the
                # interval plots all assume pred_q1 < ... < pred_qN
                if any(b <= a for a, b in zip(qs[:-1], qs[1:])):
                    exceptions.append(EpiConfigValidationError(f'Quantiles must be strictly increasing. Got {qs}'))

                # symmetric pairs: q_i + q_{N-1-i} == 1, so that each (lower, upper) pair
                # forms a central interval with nominal coverage 1 - 2 * q_i (used by WIS)
                for i in range(mid):
                    if abs(qs[i] + qs[-1 - i] - 1.0) > 1e-9:
                        exceptions.append(EpiConfigValidationError(
                            f'Quantiles must be symmetric around 0.5: {qs[i]} and {qs[-1 - i]} do not sum to 1.'))

            if self.epiconfig._num_quantiles is None:
                exceptions.append(EpiConfigValidationError(f'``_prediction_mode`` == "interval". Excpected ``_num_quantiles`` as integer but got ``None``')) 

            else:
                if self.epiconfig._num_quantiles % 2 == 0:
                    exceptions.append(EpiConfigValidationError(f'Number of quantiles must be an odd number centered around 0.5.')) 

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