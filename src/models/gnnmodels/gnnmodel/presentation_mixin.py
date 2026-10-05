from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np

from ...utils import ModelStatus
from ....dataloading.databuilders import GraphDataBuilder
from ....utils.textformatting import section, align

if TYPE_CHECKING:
    from ..utils import Strategy   
    from ...utils import PredictionManager 

class GNNModelPresentationMixin:
    """ 
    Mixin class to ``GNNModel`` that deals with the representation of the model.
    """    
    name:               str 
    model_class:        str
    status_dict:        dict[ModelStatus, bool]
    config_info:        dict[str, Any]
    strategy:           Strategy 
    predictions:        PredictionManager
    verbose:            int
    n_epochs:           int
    databuilder:  GraphDataBuilder

    def _return_verbose_iter(self) -> tuple[list[int], range | range]:
        verbose_loops : list[int] = list(np.arange(1, self.n_epochs + 1)) # type: ignore
        epoch_iter      = range(self.n_epochs)
        return verbose_loops, epoch_iter   

    def _return_verbose_line(self, 
                             epoch: int | None  = None, 
                             train_loss: float | None= None, 
                             val_loss: float | None= None, 
                             new_best: str | None  = None, 
                             patience: str | None  = None, 
                             lr_updated: bool | None = None):
        """prints a single line in the training - table"""
        columns = ["epoch", "train loss", "val loss", "new best", "patience"]
        columns = [col.upper() for col in columns]

        widths = [5, 10, 10, 8, 9]
        alignments = ["^", "^", "^", "^", "^"]

        def fmt(value: Any, width: Any, align: Any):
            return f"{value:{align}{width}}"

        def make_row(values : list[str]):
            return "| " + " | ".join(
                fmt(v, w, a) for v, w, a in zip(values, widths, alignments)
            ) + " |"

        # total table width = pipes + spaces + column widths
        total_width = sum(widths) + 3 * len(widths) + 1
        separator = "─" * total_width

        if any(x is not None for x in (epoch, train_loss, val_loss, new_best, patience, lr_updated)):
            row_values = [
                f"{epoch:03d}" if epoch is not None else "",
                f"{train_loss:.4f}" if train_loss is not None else "",
                f"{val_loss:.4f}" if val_loss is not None else "",
                f"{new_best}" if new_best is not None else "",
                f"{patience}" if patience else "",
            ]
            line = make_row(row_values)
            if lr_updated:
                line += " *"
            print(line)
        else:
            print(separator)
            print(make_row(columns))            

    def __str__(self):
        # Calculate width
        all_keys = (
            ['model name', 'model class'] +
            list(self.status_dict.keys()) +
            list(self.config_info.get('model_hparams', {}).keys()) +
            list(self.config_info.get('global_hparams', {}).keys())
        )
        width = max(len(k) for k in all_keys) if all_keys else 20
        
        # Build output
        lines = ['<DeepModel(']
        lines.append(align('model name', self.name, width))
        lines.append(align('model class', self.model_class, width))
        lines.append('')
        
        # Status section
        status_items = {k: "✓" if v else "✗" for k, v in self.status_dict.items()}
        lines.extend(section('status', status_items, width))
        lines.append('')
        
        # Forecasts section
        lines.extend(section('forecasts', {'forecasted': str(self.predictions)}, width))
        lines.append('')
        
        # Model hparams
        model_hparams = dict(self.config_info.get('model_hparams', {}))
        model_hparams['strategy'] = self.strategy
        lines.extend(section('model hparams', model_hparams, width))
        lines.append('')
        
        # Global hparams
        lines.extend(section('global hparams', self.config_info.get('global_hparams', {}), width))
        
        lines.append(')>')
        
        return '\n'.join(lines)
