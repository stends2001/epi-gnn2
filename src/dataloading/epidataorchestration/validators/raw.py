import pandas as pd 

from .base import EpiDataContainerValidator
from .exceptions import MissingColumnError
from ..containers import RawEpiData
from ...epiconfig.epiconfig import EpiConfig

class RawValidator(EpiDataContainerValidator):
    """ 
    Validates ``RawEpiData``. Validates that attributes given are of allowed type,
    and that they are not empty. Also validates that expected columns are present.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.            
    rawepidata : RawEpiData
        Data class container for raw data to be validated.    
    
    See Also
    --------
    For more information, please see the Parent class:
    ``EpiDataContainerValidator``   
    """    

    def __init__(self,
                 epiconfig : EpiConfig,
                 rawepidata : RawEpiData):

        super().__init__(epiconfig, 
                         dataclass_validated='RawEpiData')

        self.rawepidata= rawepidata
        self.cols      = ['level','key']

    def validate(self):
        attrs           = self._get_expected_attributes()
        
        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.rawepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)
        
            # validate mandatory columns being present
            if isinstance(stored_attribute, pd.DataFrame) and attr_name not in ['disease','region_harmonization']:
                 self._validate_presence_columns(attr_name, stored_attribute)
                
    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):
        for col in self.cols:
            if col not in stored_attribute:
                raise MissingColumnError(attribute_name, col, self.dataclass_validated)        

    def _get_expected_attributes(self) -> list[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['disease','population_size','shapedata','region_harmonization','tokenization_map']

        if self.epiconfig.feature_popdens:
            expected_attributes.append('population_density')
              

        return expected_attributes