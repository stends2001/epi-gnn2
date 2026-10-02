from __future__ import annotations

from typing import TYPE_CHECKING
import re

from ..utils import model_colors, ModelStatus
from ...utils import section, checkmark, crossmark

if TYPE_CHECKING:
    from ...dataloading.columnregistration import ColumnRegistry
    from ...dataloading.epiconfig import EpiConfig

class ModelAppearanceMixin:
    """ 
    Mixin class to ``BaseModel`` that deals with the model's appearance 
    (colors, name, representation)
    """
    name : str
    model_class : str
    epiconfig : EpiConfig
    column_registration : ColumnRegistry
    status_dict : dict[ModelStatus, bool]
    model_class : str 

    def _get_model_color(self) -> str:
        """returns model-color in string format based on the lookup in model_colors"""
        lookup_name = self.model_class.lower()
        
        if lookup_name not in model_colors:
            raise ValueError(f'no color set for model of class {lookup_name}')
        
        else:
            return model_colors[lookup_name]
        
    def _get_clean_name(self) -> str:
        """Return a filesystem-safe version of the model name."""
        name = self.name.lower()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^a-z0-9_-]", "", name)
        name = re.sub(r"_+", "_", name)
    
        return name.strip("_")

    def _print_status_update(self, status: ModelStatus):
        """Print status update depending on `verbose`"""

        if status == "model_initialized":
            self._print_header()
        else:
            print(f"{status} {checkmark}")

    def _print_header(self):
        """Print formatted model header."""

        total_width     = 50
        title           = self.name

        inner_width     = total_width - 4
        centered_title  = title.center(inner_width)

        print(
            "\n"
            + "=" * total_width + "\n"
            + f"=={centered_title}==\n"
            + "=" * total_width
        )

    def __repr__(self) -> str:
        """minimal representation"""
        return f"{self.__class__.__name__}(name={self.name!r})"    

    def __str__(self) -> str:
        """extensive representation"""
        all_keys = (
            ['name', 'model_class'] + list(self.status_dict.keys())
        )

        width = max(len(k) for k in all_keys) if all_keys else 20
        
        # Build output
        lines = [f'<{self.__class__.__name__}(']
        lines.append('')
        general_items = {'name': self.name, 'model_class': self.model_class}
        lines.extend(section('generics', general_items, width))
        lines.append('')
        
        # Status section
        status_items = {k: f"{checkmark}" if v else f"{crossmark}" for k, v in self.status_dict.items()}
        lines.extend(section('status', status_items, width))
        lines.append('')
        
        lines.append(')>')
        
        return '\n'.join(lines)