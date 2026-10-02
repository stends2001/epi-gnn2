from __future__ import annotations

from typing import assert_never, TYPE_CHECKING
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime

from ..utils import color_is_light, SingleNodeType
from ...dataloading import GraphDataBuilder, BaseLineDataBuilder
from ...utils import DataSetSplit, testcolor

if TYPE_CHECKING:
    from ..utils import PredictionManager
    from ...dataloading import ColumnRegistry
    from ...dataloading import EpiConfig
    from ...dataloading.epidataorchestration.containers import ContextEpiData
    from ...dataloading.epidataorchestration.utils.temporal_summary import EpiDataTemporalSummary

class ForecastDisplayMixin:
    """ 
    Mixin class to ``BaseModel`` that deals with the visualizing the model's 
    predictions.    
    Contains a single public method, `show_forecats()` with
    supportive hidden methods.
    """
    predictions : PredictionManager
    databuilder : GraphDataBuilder | BaseLineDataBuilder
    epiconfig : EpiConfig
    column_registration : ColumnRegistry
    context_data : ContextEpiData
    temporal_summary : EpiDataTemporalSummary
    model_color : str
    name : str

    # ======== PUBLIC METHODS ========= #
    def show_forecasts(self,
                       node_idx : SingleNodeType | list[SingleNodeType] = 0,
                       dataset : DataSetSplit = 'test',
                       horizon : int = 0,
                       is_original: bool = True,
                       ) -> tuple[Figure, list[Axes]]:
        """
        Plot model's forecasts

        Parameters
        ----------
        node_idx : SingleNodeType | list[SingleNodeType]
            Nodes for which to plot the forecasts. May also include ``'national'``, 
            otherwise integers.
        dataset: DataSetSplit = 'test'
            Dataset (``train``, ``val`` or ``test``) to show predictions for.
        horizon : int = 0            
            The horizon (index) to show predictions for. NOTE these always start at 0.
        is_original : bool = True
            Whether to show original-scale predictions (``True``) or transformed-scale 
            (``False``).
        """       
        # ==== get constants ===== #
        x_range             = self.temporal_summary.get_daterange_dataset(dataset, reference = 'target')       

        # extend axes to one step before and one after the first and last pred      
        xlimits             = [self.temporal_summary._shift(x_range[0], -1), self.temporal_summary._shift(x_range[1], 1)]        
        timesteps_ahead     = int(self.databuilder.dataorchestrator.config.horizon_leadtime + horizon)       
       
        # ==== get predictions ==== #
        nodes_list, df_pred, df_pred_aggr = self._get_forecast_dfs(node_idx, dataset, horizon, is_original)

        # ===== plot ===== #
        n_plots = len(nodes_list)
        fig, axes_array = plt.subplots(n_plots, 1, figsize=(16, 2 + 5 * n_plots), squeeze=False)
        axes: list[Axes]= list(axes_array.flatten())

        for plot_idx, id in enumerate(nodes_list):
            ax = axes[plot_idx]

            # ======== get pred data ========== #
            match id:

                case 'national':
                    if df_pred_aggr is None:
                        raise ValueError('Missing df_aggregated in show_forecasts!')
                    
                    df_node     = df_pred_aggr 
                    ax_title    = f'nationally aggregated'

                case int():
                    if df_pred is None:
                        raise ValueError(f'Missing df in show_forecasts for nodes!')                    
                    
                    df_node     = df_pred[df_pred[self.epiconfig.id_column] == id]
                    nodename    = self.context_data.nodenames[self.context_data.nodenames[f'{self.epiconfig.id_column}'] == id][f'{self.epiconfig.level}_name'].iloc[0]
                    ax_title    = f'{nodename} [{self.epiconfig.id_column} {id}]'

                case _:
                    assert_never(id)
                        
            # draw target
            self._draw_target_on_ax(df_node, ax)
        
            # draw predictions
            self._draw_preds_on_ax(df_node, ax)            

            # basic make up of ax
            self._make_up_ax(xlimits, ax_title, ax)
            
        suptitle = self._return_suptitle(dataset, timesteps_ahead, is_original)
    
        plt.close()
            
        fig.suptitle(suptitle, fontweight = 'bold', fontsize = 14)
        return fig, axes

    # ======== HIDDEN METHODS ========= #
    def _get_forecast_dfs(self, 
                          node_idx : SingleNodeType | list[SingleNodeType], 
                          dataset : DataSetSplit, 
                          horizon : int, 
                          is_original : bool
                          ) -> tuple[list[SingleNodeType], pd.DataFrame | None, pd.DataFrame | None]:
        """Retrieves the prediction - datasets for the given input and returns those."""
        nodes_list: list[SingleNodeType]

        predictioncollection = self.predictions.get_preds(dataset)

        df_pred_aggr    = None  
        df_pred         = None

        match node_idx:

            case int():
                nodes_list = [node_idx]
                df_pred    = predictioncollection.get(horizon, is_original, False)

            case str() if node_idx == 'national':
                nodes_list   = [node_idx]
                df_pred_aggr = predictioncollection.get(horizon, True, True)

            case str():
                raise ValueError(f'only "national" is valid as string node_idx, got {node_idx}')

            case list():
                nodes_list = node_idx
                if 'national' in nodes_list:
                    df_pred_aggr = predictioncollection.get(horizon, True, True)

                if any(n != 'national' for n in nodes_list):
                    df_pred      = predictioncollection.get(horizon, is_original, False)

            case _:
                assert_never(node_idx)
            
        ints = [x for x in nodes_list if isinstance(x, int)]

        for node_idx in ints:
            if node_idx > (self.context_data.num_nodes - 1):
                raise ValueError(
                    f'node {node_idx} invalid. There are {self.context_data.num_nodes} nodes [0-{self.context_data.num_nodes-1}]'
                    ) 

        return nodes_list, df_pred, df_pred_aggr
    
    def _draw_preds_on_ax(self, df_node: pd.DataFrame, ax: Axes):
        """plots predictions (single line for point preds) on given ax."""
        # single line: center col
        sns.lineplot(data           = df_node, 
                    x               = self.epiconfig.temporal_column, 
                    y               = self.epiconfig.pred_column, 
                    color           = self.model_color, 
                    marker          = 'o', 
                    label           = 'point predictions',
                    ax              = ax, 
                    linewidth       = 2, 
                    markeredgewidth = 0.3,

                # adjust some color-aspects if color is light
                markeredgecolor = 'black' if color_is_light(self.model_color) else 'white',
                )        

    def _draw_target_on_ax(self, df_node: pd.DataFrame, ax: Axes):
        """plots targets in single line on given ax."""        
        sns.lineplot(data       = df_node, 
                    x          = self.epiconfig.temporal_column, 
                    y          = 'target',    
                    color      = testcolor,          
                    marker     = 'o',   
                    label      = 'ground truth', 
                    ax         = ax, 
                    linewidth  = 2)        
    
    def _make_up_ax(self, xlimits: list[datetime], ax_title: str, ax: Axes):
        """sets basic ax-layout"""
        ax.set_xlabel("")   
        ax.set_xlim(xlimits)    # xlimits is a list of two datetime objects # type: ignore
        ax.set_title(ax_title)    
        ax.legend()
        ax.grid()          

    def _return_suptitle(self, dataset: DataSetSplit, timesteps_ahead: int, is_original: bool) -> str:
        """returns suptitle based on plotting-function's input."""
        suptitle = f'{self.databuilder.dataorchestrator.config.target_column} predictions by {self.name}, {timesteps_ahead}{self.databuilder.dataorchestrator.config.temporal_frequency} ahead' 

        if dataset != 'test':
            suptitle += f" [{dataset}]"

        if not is_original:
            suptitle += ' [transformed]'           

        return suptitle 