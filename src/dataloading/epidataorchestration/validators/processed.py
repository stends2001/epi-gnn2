import pandas as pd 

from .base import EpiDataContainerValidator
from .exceptions import MissingColumnError, NaNsFoundError
from ..containers import ProcessedEpiData
from ...epiconfig import EpiConfig

class ProcessedValidator(EpiDataContainerValidator):
    """ 
    Validates ``ProcessedEpiData``. Validates that attributes given are of allowed type,
    and that they are not empty. Also validates there's no NaNs, and that expected
    columns are present.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.            
    processedepidata : ProcessedEpiData
        Data class container for processed data to be validated.

    See Also
    --------
    For more information, please see the Parent class:
    ``EpiDataContainerValidator``   
    """    

    def __init__(self,
                 epiconfig : EpiConfig,
                 processedepidata : ProcessedEpiData):
        
        super().__init__(epiconfig, 
                         dataclass_validated='ProcessedEpiData')

        self.processedepidata= processedepidata
        self.col = self.epiconfig.id_column

    def validate(self):
        attrs           = self._get_expected_attributes()

        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.processedepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)
                           
            if isinstance(stored_attribute, pd.DataFrame):
                
                # validate mandatory columns being present                
                self._validate_presence_columns(attr_name, stored_attribute)

                # validate absence of NaNs
                self._validate_nan(attr_name, stored_attribute)
               
    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):
        """validates that required columns are present."""        

        if self.col not in stored_attribute:
            raise MissingColumnError(attribute_name, self.col, self.dataclass_validated)        
        
    def _validate_nan(self, attribute_name: str, stored_attribute: pd.DataFrame):
        """validates that there's no NaNs, anywhere in any column."""

        nan_columns = stored_attribute.columns[stored_attribute.isna().any()].tolist()
        if nan_columns:     
            NaNsFoundError(attribute_name, self.dataclass_validated, nan_columns)
        
    def _get_expected_attributes(self) -> list[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['epidata']

        if self.epiconfig.feature_popdens:
            expected_attributes.append('population_density')

        return expected_attributes