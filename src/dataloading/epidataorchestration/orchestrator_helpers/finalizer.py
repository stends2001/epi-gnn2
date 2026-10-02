import time
import pandas as pd

from ...columnregistration.exceptions import ColEntryMissingAttribute

from ...epiconfig import EpiConfig

from ..containers import TransformedEpiData, FinalizedEpiData
from ..utils.normalization import reverse_log, reverse_minmax, reverse_zscore
from ...columnregistration import ColumnRegistry

class EpiDataFinalizer:
    """
    ``EpiDataOrchestrator`` utility class that creates the ``FinalizedEpiData``. 
    ``EpiDataFinalizer`` finalizes the normalized data set, and creates a reverse-
    normalized data frame too.
    
    Besides a handful of helper methods, ``EpiDataFinalizer`` has 
    an ``orchestrate()`` method, which returns the ``FinalizedEpiData``.
    
    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.    
    column_registration : ColumnRegistry
        Registry of columns that keeps track of all columns and transformations.

    See Also
    --------
    ``ColumnRegistry``
        Stores all columns and transformations. This starts in ``EpiFeatureBuilder``,
        and is built upon further in EpiDataTransformer.

    Downstream
    ----------        
    ``EpiDataOrchestrator`` has six utility classes, each of which is responsible
    for a single stage in the pipeline of getting model-ready datasets. 
    ``EpiDataFinalizer`` is the sixt, and last one.    
    """   
    def __init__(self, 
                 epiconfig : EpiConfig, 
                 column_registration : ColumnRegistry):
        
        self.epiconfig = epiconfig 
        self.column_registration = column_registration

    def _create_pred_col_entry(self):
        """
        while pred doesn't exist in the data, models will end up with these columns.
        if prediction_quantiles are inputted in EpiConfig, these will be created here.
        """

        needs_normalization  = False if self.epiconfig.target_column == 'cases' else True
        transformation_group = 'target'if self.epiconfig.target_column != 'cases' else None

        self.column_registration.add_column(
            'pred',
            'pred',
            transformation = needs_normalization,
            transformation_group = transformation_group
        )                   

    def _add_horizons(self, df: pd.DataFrame) -> pd.DataFrame:
        """adds target columns when horizon_size>1"""
        base_lead = self.epiconfig.horizon_leadtime
        for additional_steps in range(0, self.epiconfig.horizon_size):
            steps_ahead = base_lead + additional_steps
            target_col = f'target_lead{steps_ahead}'
            
            # Shift from the base target
            df[target_col] = df.groupby(self.epiconfig.id_column)[f'target'].shift(-additional_steps)
            
            needs_normalization  = False if self.epiconfig.target_column == 'cases' else True
            transformation_group = 'target'if self.epiconfig.target_column != 'cases' else None

            # Register in column registry
            self.column_registration.add_column(
                target_col,
                'target',
                transformation=needs_normalization,
                transformation_group=transformation_group
            )
           
        return df.drop(columns = ['target'])

    def _drop_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """drop nans"""

        return df.dropna()

    def _denormalize(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
        """Reverse all transformations in reverse order of application: norm first, then log."""
        dfc = normalized_df.copy()

        if not self.epiconfig.normalization_method:
            return dfc

        for col_entry in self.column_registration._entries:
            # TODO: MAKE MATCH/CASE
            if not col_entry.transformation:
                continue

            if col_entry.column_name not in dfc.columns:
                continue
            
            if col_entry.transformation_group == 'self':
                params = col_entry.transformation_params

                if params is None:
                    raise ColEntryMissingAttribute(
                        col_entry.column_name, "transformation_params"
                        )   

            elif col_entry.transformation_group is None:
                raise ColEntryMissingAttribute(
                    col_entry.column_name, "transformation_group"
                    )
                               
            else:
                ref    = self.column_registration.get_entry_by_name(col_entry.transformation_group)
                params = ref.transformation_params

            if params is None:
                continue

            if params.zscore is not None:
                dfc = reverse_zscore(dfc, col_entry.column_name, params.zscore)
            elif params.minmax is not None:
                dfc = reverse_minmax(dfc, col_entry.column_name, params.minmax)

            if params.log is not None:
                dfc = reverse_log(dfc, col_entry.column_name, params.log)

        return dfc

    def orchestrate(self, normalized_data: TransformedEpiData) -> 'FinalizedEpiData':
        time_start = time.time()
        dfc         = normalized_data.data

        dfc         = self._add_horizons(dfc)
        self._create_pred_col_entry()
    
        dfc_normalized_nanfree      = self._drop_nans(dfc)
        dfc_denormalized_nanfree    = self._denormalize(dfc_normalized_nanfree) 

        time_end = time.time()

        time_elapsed = time_end - time_start

        return FinalizedEpiData(
            data        = dfc_normalized_nanfree,
            data_denorm = dfc_denormalized_nanfree,
        )