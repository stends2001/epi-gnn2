from typing import assert_never
import pandas as pd 

from .base import EpiDataContainerValidator
from .exceptions import MissingColumnError, NaNsFoundError, InvalidNormalizationError
from ..containers import TransformedEpiData
from ...columnregistration import ColumnRegistry
from ...epiconfig import EpiConfig
from ...columnregistration.exceptions import ColEntryMissingAttribute

class TransformedValidator(EpiDataContainerValidator):
    """ 
    Validates ``TransformedEpiData``. Validates that attributes given are of allowed 
    type, and that they are not empty. Also validates that expected columns are present,
    no NaNs present, and that the normalization process went well (goes over 
    normalization parameters).

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.    
    column_registration : ColumnRegistry
        Registry of columns that keeps track of all columns and transformations.                   
    normalizedepidata : TransformedEpiData
        Data class container for transformed data to be validated.      

    See Also
    --------
    For more information, please see the Parent class:
    ``EpiDataContainerValidator``   
    """ 
    
    def __init__(self,
                 epiconfig : EpiConfig,
                 column_registry : ColumnRegistry,                 
                 normalizedepidata : TransformedEpiData):

        super().__init__(epiconfig, 
                         dataclass_validated='TransformedEpiData')

        self.normalizedepidata= normalizedepidata
        self.column_registry  = column_registry
        self.cols             = [c.column_name for c in column_registry._entries]

    def validate(self):
        """
        """
        attrs           = self._get_expected_attributes()

        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.normalizedepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)
                           
            # validate mandatory columns being present                
            self._validate_presence_columns(attr_name, stored_attribute)

            # validate absence of NaNs
            self._validate_nan(attr_name, stored_attribute)

            # validate normalization 
            self._validate_normalization(attr_name, stored_attribute)

    def _validate_normalization(self, attribute_name: str, stored_attribute: pd.DataFrame):
        """
        Validates the normalization process. Thus, when minmax, that min = 0 and max = 1. 
        When zscore, that mean is 0 and std = 1.
        """
        tolerance = 1e-6
        train_df  = stored_attribute[stored_attribute['train']]

        for col_entry in self.column_registry._entries:

            match (col_entry.transformation, col_entry.transformation_group):

                case (False, _):
                    continue

                case (True, 'self'):
                    params = col_entry.transformation_params

                case (True, str()):
                    # referral columns — params are validated via their reference column
                    continue

                case (True, None):
                        raise ColEntryMissingAttribute(
                            col_entry.column_name, "transformation_params"
                            )    
                case _:
                    assert_never(col_entry.transformation)

            if params is None:
                continue

            if params.zscore is not None:
                computed_mean = train_df[col_entry.column_name].mean()
                computed_std  = train_df[col_entry.column_name].std()

                if abs(computed_mean) > tolerance:
                    raise InvalidNormalizationError(
                        attribute_name, self.dataclass_validated, col_entry.column_name,
                        f"Expected mean ≈ 0 (tolerance {tolerance}). Got {computed_mean}"
                    )
                if abs(computed_std - 1.0) > tolerance:
                    raise InvalidNormalizationError(
                        attribute_name, self.dataclass_validated, col_entry.column_name,
                        f"Expected std ≈ 1.0 (tolerance {tolerance}). Got {computed_std}"
                    )

            elif params.minmax is not None:
                computed_min = train_df[col_entry.column_name].min()
                computed_max = train_df[col_entry.column_name].max()

                if abs(computed_min) > tolerance:
                    raise InvalidNormalizationError(
                        attribute_name, self.dataclass_validated, col_entry.column_name,
                        f"Expected min ≈ 0 (tolerance {tolerance}). Got {computed_min}"
                    )
                if abs(computed_max - 1.0) > tolerance:
                    raise InvalidNormalizationError(
                        attribute_name, self.dataclass_validated, col_entry.column_name,
                        f"Expected max ≈ 1.0 (tolerance {tolerance}). Got {computed_max}"
                    )
                       
    def _validate_presence_columns(self, attribute_name: str, stored_attribute: pd.DataFrame):
        """validates that required columns are present."""              
        for col in self.cols:
            if col not in stored_attribute.columns:
                raise MissingColumnError(attribute_name, col, self.dataclass_validated)        
        
    def _validate_nan(self, attribute_name: str, stored_attribute: pd.DataFrame):
        """validates that there's no NaNs, anywhere in any column."""        
        
        nan_columns = stored_attribute.columns[stored_attribute.isna().any()].tolist()
        if nan_columns:     
            NaNsFoundError(attribute_name, self.dataclass_validated, nan_columns)
        
    def _get_expected_attributes(self) -> list[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['data']       

        return expected_attributes