from __future__ import annotations

from typing import Any, TYPE_CHECKING, Literal
import pandas as pd
import torch 
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..utils import LossManager
from ...utils import ModelStatus
from ....dataloading import GraphDataBuilder
from ....utils import traincolor, valcolor

if TYPE_CHECKING:
    from ....dataloading.epiconfig import EpiConfig
    from ..utils import Strategy    

Metric = Literal['train_loss', 'val_loss', 'learning_rate']


class GNNModelTrainMixin:
    """ 
    Mixin class to ``GNNModel`` that deals with training of models.    
    """    
    status_dict:        dict[ModelStatus, bool]
    epiconfig:          EpiConfig
    config_info:        dict[str, Any]    
    model:              torch.nn.Module
    databuilder:  GraphDataBuilder
    strategy:           Strategy
    device:             torch.device
    optimizer:          Optimizer
    scheduler:          _LRScheduler    
    loss:               LossManager
    min_delta:          float
    patience:           int 
    verbose:            int
    monitoring_metrics: pd.DataFrame
    n_epochs:           int

    def train(self):
        """ 
        Train model. No arguments needed, but ``model_hparams_set()`` and 
        ``global_hparams_set()`` both need to have been called.
        """
        self._check_status(['model_hparams_set', 'global_hparams_set'])
  
        train_loader = self.databuilder.dataloader_train 
        val_loader   = self.databuilder.dataloader_val 

        verbose_loops, epoch_iter = self._return_verbose_iter()

        # ====== PRE-TRAINING ====== #
        self.model.train()
        best_val_loss       = float('inf')
        patience_counter    = 0
        best_model_state    = None

        list_val_loss : list[float]       = []
        list_train_loss : list[float]     = []
        list_patience : list[bool]        = []
        list_lr : list[float]             = []

        L_train             = len(train_loader)
        L_val               = len(val_loader)

        self._return_verbose_line()

        # Each epoch is divided into:
        #   1. training phase
        #   2. validation phase
        #   3. update phase
        
        for epoch in epoch_iter:
            # for printing purposes
            repr_epoch = epoch + 1 

            # ======================== TRAINING PHASE ========================
            total_loss = 0
            
            for snapshot in train_loader:
                snapshot = snapshot.to(self.device)

                # different models have different input and output in steps
                # taken care of using the strategy

                loss_train = self.strategy.training_step(
                    model       = self.model, 
                    snapshot    = snapshot, 
                    optimizer   = self.optimizer, 
                    loss_fn     = self.loss
                )
                
                total_loss += loss_train
            
            train_loss = total_loss / L_train
            list_train_loss.append(train_loss)

            # ======================== VALIDATION PHASE ========================
            self.model.eval()
            val_loss = 0
            

            with torch.no_grad():
                for snapshot in val_loader:
                    snapshot = snapshot.to(self.device)

                    loss_val = self.strategy.validation_step(
                        model       = self.model, 
                        snapshot    = snapshot, 
                        loss_fn     = self.loss
                    )

                    val_loss += loss_val
            
            val_loss = val_loss / L_val
            list_val_loss.append(val_loss)
            
            # ======================== UPDATE PHASE ========================
            # the lr used in this epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            list_lr.append(current_lr)
            
            self.model.train()

            # Check if validation loss improved
            val_improved = val_loss < (best_val_loss - self.min_delta)

            # if so => save best model
            if val_improved:
                best_val_loss   = val_loss
                patience_counter= 0
                best_model_state= self.model.state_dict().copy()
                list_patience.append(False)

            else:
                patience_counter += 1
                list_patience.append(True)

            if patience_counter >= self.patience:
        

                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
              

                break              

            # Step scheduler => scheduler.step requires val loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss) # type: ignore
            # for other schedulers, no arguments required
            else:
                self.scheduler.step()

            new_lr = self.optimizer.param_groups[0]['lr']
            
            _ = self._return_verbose_line(repr_epoch, train_loss, val_loss,
                                          "v" if val_improved else None,
                                          None if val_improved else f"{patience_counter}/{self.patience}",
                                          True if current_lr != new_lr else None
                                          ) 
                     
        self.monitoring_metrics = pd.DataFrame({'train_loss'    : list_train_loss,
                                                'val_loss'      : list_val_loss,
                                                'patience'      : list_patience,
                                                'learning_rate' : list_lr}).reset_index(names='epoch') # index starting from 0
        
        self.monitoring_metrics['epoch'] = self.monitoring_metrics['epoch'] + 1
        self._update_status('trained')

    def show_monitoring_metrics(self) -> None:
        """Shows plot of trainloss, valloss, patience and learning rate per epoch."""
 
        if not hasattr(self, 'monitoring_metrics'):
            raise ValueError('no monitoring metrics found')
      
        fig, axes_array = plt.subplots(1, 3, figsize=(24, 4))
        axes: list[Axes]= list(axes_array.flatten())

        plots: list[tuple[Metric, str]] = [
            ('train_loss', traincolor),
            ('val_loss', valcolor),
            ('learning_rate', 'black'),
        ]

        for ax, (variable, color) in zip(axes, plots):
            sns.lineplot(
                data=self.monitoring_metrics,
                x='epoch',
                y=variable,
                color=color,
                label=variable.replace("_", " ").title(),
                zorder=2,
                ax=ax
            )

            self._plot_scatter_patience_on_ax(variable, ax)
            self._draw_best_epoch(variable, ax)                

        for ax in axes:
            self._format_ax(ax)

        axes[0].set_title('Training loss')      
        axes[1].set_title('Validation loss')    
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')

        fig.show()

    def _plot_scatter_patience_on_ax(self, y: Literal['train_loss','val_loss','learning_rate'], ax: Axes) -> None:
        """plots red cross for patience epoch on any ax"""
        patience_mask = self.monitoring_metrics['patience'] > 0

        ax.scatter(x        = self.monitoring_metrics['epoch'][patience_mask], 
                   y        = self.monitoring_metrics[y][patience_mask], 
                   color    ='red', 
                   marker   ='x', 
                   label    ='Patience Epochs',
                   zorder   = 1)

    def _draw_best_epoch(self, y: Literal['train_loss','val_loss','learning_rate'], ax: Axes) -> None:
        """plots a black dot and arrow on any ax for the best-epoch"""
        fraction_dist_point_arrow   = 0.1

        best_idx        = self.monitoring_metrics['val_loss'].idxmin()
        x_point, y_point= self.monitoring_metrics['epoch'][best_idx],  self.monitoring_metrics[y][best_idx]
        y_arrow         = y_point + fraction_dist_point_arrow * y_point if y != 'learning_rate' else y_point - fraction_dist_point_arrow * y_point

        ax.annotate(
                '',
                xy          =(x_point, y_arrow),
                xytext      = (0, 50) if y != 'learning_rate' else (0, -50),
                textcoords  = 'offset points',
                ha          = 'center',
                arrowprops  = dict(arrowstyle='->', color='black', lw=1.5),
                zorder      = 2
            )
        
        ax.scatter(x        = x_point, 
                   y        = y_point, 
                   color    ='black', 
                   marker   ='o', 
                   label    ='Best Epochs',
                   zorder   = 4,
                   s        = 15
                   )   

    def _format_ax(self, ax: Axes) -> None:
        """basic ax format"""
        ax.grid()
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend()

    # ======== STUBS ======= #
    def _check_status(self, required_states: list[ModelStatus] | ModelStatus) -> None: ...
    def _update_status(self, status: ModelStatus) -> None: ...
    def _return_verbose_iter(self) -> tuple[list[int], range]: ...
    def _return_verbose_line(self, 
                             epoch:         int | None  = None, 
                             train_loss:    float| None= None, 
                             val_loss:      float| None= None, 
                             new_best:      str| None  = None, 
                             patience:      str| None  = None, 
                             lr_updated:    bool| None = None): ...
