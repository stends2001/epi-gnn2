from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

from .exceptions import UnexpectedAttributeTypeError, EmptyAttributeTypeError
from ...epiconfig.epiconfig import EpiConfig

class EpiDataContainerValidator(ABC):
    """ 
    Parent class to all data - containers validators. Each such container
    is associated with one specific validator.

    Subclasses may have any helper methods, but must have one centralized ``validate()`` 
    method that orchestrates and runs the validation, analogous to the 
    ``orchestratre()`` method on the ``EpiDataOrchestrator`` helper classes.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.            
    dataclass_validated : str
        Name of dataclass to be validated.
    
    Methods
    -------
    Some methods are shared and therefore defined in this parent class. This includes:
    - `validate_type()`
    - `validate_length_nonzero()`
    
    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. Each of these intermediate containers is associated
    with a single child-class of ``EpiDataContainerValidator``.    

    Ultimately, when the ``EpiDataOrchestrator`` is done, the final data will be fed
    into the model-specific databuilder (basically a dataloader) class.
    """
    def __init__(self,
                epiconfig : EpiConfig,
                dataclass_validated : str):
        
        self.epiconfig =  epiconfig
        self.dataclass_validated = dataclass_validated

        self.allowed_types   = (pd.DataFrame, dict)

    @abstractmethod
    def validate(self):
        """abstractmethod; all child classes must implement this method."""
        pass

    def _validate_type(self, attribute_name: str, stored_attribute: Any):
        """validates type of attribute. Must be one of `self.allowed_types`."""
        if not isinstance(stored_attribute, self.allowed_types):

            raise UnexpectedAttributeTypeError(attribute_name, 
                                               self.dataclass_validated, 
                                               str(type(stored_attribute)), 
                                               [str(obj) for obj in self.allowed_types])
        
    def _validate_length_nonzero(self, attribute_name: str, stored_attribute: pd.DataFrame | dict):
        """validates that attribute is not emtpy"""
        if not len(stored_attribute) > 0:
            raise EmptyAttributeTypeError(attribute_name, self.dataclass_validated)
                    