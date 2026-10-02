import pandas as pd 

from .base import EpiDataContainerValidator
from .exceptions import MissingColumnError, InvalidTokenizationError
from ..containers import HarmonizedEpiData
from ...epiconfig import EpiConfig

class HarmonizedValidator(EpiDataContainerValidator):
    """ 
    Validates ``HarmonizedEpiData``. Validates that attributes given are of allowed type,
    and that they are not empty. Also validates there's no NaNs, and that expected
    columns are present.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.            
    harmonziedepidata : HarmonizedEpiData
        Data class container for harmonized data to be validated.

    See Also
    --------
    For more information, please see the Parent class:
    ``EpiDataContainerValidator``   
    """

    def __init__(self,
                 epiconfig : EpiConfig,
                 harmonziedepidata : HarmonizedEpiData):

        super().__init__(epiconfig, 
                         dataclass_validated='HarmonizedEpiData')

        self.harmonziedepidata= harmonziedepidata
        self.required_col = self.epiconfig.id_column

    def validate(self):
        attrs           = self._get_expected_attributes()

        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.harmonziedepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)
                           
            if isinstance(stored_attribute, pd.DataFrame):
                # validate mandatory columns being present
                self._validate_presence_columns(attr_name, stored_attribute)

                # validate tokenizatino
                self._validate_tokenization(attr_name, stored_attribute)
               
    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):
        """validates that the only required column is present everywhere"""
        if self.required_col not in stored_attribute:
            raise MissingColumnError(attribute_name, self.required_col, self.dataclass_validated)        

    def _validate_tokenization(self, attribute_name: str, df: pd.DataFrame):
        """checks any missing values within the tokenization"""
        col             = df[self.epiconfig.id_column].unique()
        unique_tokens   = sorted(set(int(x) for x in col))

        min_val         = unique_tokens[0]
        max_val         = unique_tokens[-1]

        expected_set    = set(range(min_val, max_val + 1))

        actual_set      = set(unique_tokens)

        missing_values = sorted(expected_set - actual_set)
        leftover_values= sorted(actual_set - expected_set)

        if len(missing_values) > 0 or len(leftover_values) > 0:
            raise InvalidTokenizationError(attribute_name, self.dataclass_validated, missing_values, leftover_values)

    def _get_expected_attributes(self) -> list[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['epidata']

        # the following are optional, depending on epiconfig
        if self.epiconfig.feature_popdens:
            expected_attributes.append('population_density')     

        return expected_attributes