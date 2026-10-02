from __future__ import annotations

from .base import EpiDataContainerValidator
from ..containers import ContextEpiData
from ...epiconfig import EpiConfig

class ContextValidator(EpiDataContainerValidator):
    """ 
    Validates ``ContextEpiData``. Validates that attributes given are of allowed type,
    and that they are not empty.

    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.            
    contextepidata : ContextEpiData
        Data class container for context data to be validated.

    See Also
    --------
    For more information, please see the Parent class:
    ``EpiDataContainerValidator``   
    """
    def __init__(self,
                 epiconfig : EpiConfig,
                 contextepidata : ContextEpiData):

        super().__init__(epiconfig, 
                         dataclass_validated='ContextEpiData')

        self.contextepidata= contextepidata
        self.col              = self.epiconfig.id_column

    def validate(self):
        attrs           = self._get_expected_attributes()

        for attr_name in attrs:

            # retrieve attribute
            stored_attribute = getattr(self.contextepidata, attr_name)

            # validate the type
            self._validate_type(attr_name, stored_attribute)

            # validate the size 
            self._validate_length_nonzero(attr_name, stored_attribute)

    def _get_expected_attributes(self) -> list[str]:
        """returns a list of strings with expected attributes"""
        # these are mandatory
        expected_attributes = ['global_shapedata',
                               'local_shapedata',
                               'population_size',
                               'nodenames',
                               'region_harmonization',
                               'tokenization_map']
        return expected_attributes