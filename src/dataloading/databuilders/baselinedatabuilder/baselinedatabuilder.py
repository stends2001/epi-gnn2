import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...epidataorchestration.orchestrator import EpiDataOrchestrator

class BaseLineDataBuilder:
    """
    Dataloader for baselinemodels, based on 
    dataorchestrator.data_final.data_denorm

    runs by itself from init, no need to call any method

    Parameters
    ----------
    dataorchestrator: 'EpiDataOrchestrator'

    See Also
    --------
    For more info, please check EpiDataOrchestrator.

    Downstream
    ----------
    This is the dataloader - meant for BaseLineModels
    """
    def __init__(self, 
                 dataorchestrator: 'EpiDataOrchestrator'):
        
        self.dataorchestrator     = dataorchestrator
        self.column_registration  = dataorchestrator.column_registration
        
    def build(self):
        """writes the dataloader. main functionality of this class"""
        main_data           = self.dataorchestrator.data_final.data_denorm.copy()
        
        split_colnames      = self.dataorchestrator.column_registration.get_entries_names_by_type('split')
        time_colname        = self.dataorchestrator.config.temporal_column
        id_colname          = self.dataorchestrator.config.id_column
        
        # Get target from the first horizon
        base_lead           = self.dataorchestrator.config.horizon_leadtime
        target_colname      = f'target_lead{base_lead}'
        
        # Get the split columns from normalized data (splits are the same)
        timesplits          = self.dataorchestrator.data_final.data[[time_colname] + split_colnames].drop_duplicates().reset_index(drop=True)
        
        # Extract what we need from original data
        main_data_selection = main_data[[time_colname, id_colname, target_colname]]
        main_data_selection = main_data_selection.rename(columns={target_colname: 'target'})
        
        # Merge with splits
        main_data_selection = pd.merge(main_data_selection, timesplits, on=time_colname)

        self.dataloader_main = main_data_selection
        return self

    def __repr__(self):
        return (f"<BaseLineDataLoaderManager(dataloader at .dataloader_main)>")            