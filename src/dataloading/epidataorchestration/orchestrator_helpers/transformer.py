import time
from typing import assert_never
import pandas as pd

from ...columnregistration.exceptions import ColEntryMissingAttribute

from ...epiconfig import EpiConfig

from ..utils.normalization import (
    compute_zscore_params, compute_minmax_params,
    apply_log, apply_zscore, apply_minmax
)
from ...columnregistration import (
    LogParams,
    TransformationParams, ColumnRegistry, ColEntry
)

from ..containers import FeatureEpiData, TransformedEpiData
from ..utils import EpiDataTemporalSummary,  EpiDataOrchestrationError

class EpiDataTransformer:
    """
    ``EpiDataOrchestrator`` utility class that creates the ``TransformedEpiData``. 
    ``EpiDataTransformer`` transforms the feature data, by adding splits 
    (train/val/test) and normalizing and log-transforming data when necessary.
    
    Besides a handful of helper methods, ``EpiDataTransformer`` has 
    an ``orchestrate()`` method, which returns the ``TransformedEpiData``.
    
    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.    
    column_registration : ColumnRegistry
        Registry of columns that keeps track of all columns and transformations.
    temporal_summary : EpiDataTemporalSummary
        Helper class that stores temporal information, built based on ``EpiConfig``.

    See Also
    --------
    ``ColumnRegistry``
        Stores all columns and transformations. This starts in ``EpiFeatureBuilder``,
        and is built upon further in EpiDataTransformer.

    Downstream
    ----------        
    ``EpiDataOrchestrator`` has six utility classes, each of which is responsible
    for a single stage in the pipeline of getting model-ready datasets. 
    ``EpiFeatureBuilder`` is the fifth one.    
    """    
    def __init__(self,
                 epiconfig : EpiConfig,
                 column_registration : ColumnRegistry,
                 temporal_summary : EpiDataTemporalSummary):

        self.epiconfig           = epiconfig
        self.temporal_summary    = temporal_summary
        self.column_registration = column_registration

    def _set_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        set split columns:
        - ``'train'``
        - ``'val'``
        - ``'test'``
        """
        splits = self.temporal_summary.get_target_splits()

        df['train'] = df[self.epiconfig.temporal_column] < splits['trainval']
        df['val']   = (df[self.epiconfig.temporal_column] >= splits['trainval']) & \
                      (df[self.epiconfig.temporal_column] < splits['valtest'])
        df['test']  = df[self.epiconfig.temporal_column] >= splits['valtest']

        for split_col in ['train', 'val', 'test']:
            self.column_registration.add_column(split_col, 'split')

        return df

    # ── pass 1: register + apply log ─────────────────────────────────────

    def _update_columnregistry_log_params(self) -> None:
        """Register LogParams for all columns that need log transform."""
        if not self.epiconfig.log_transform:
            return

        log_params = LogParams(shift=self.epiconfig.log_shift)

        for col in self.epiconfig.log_transform:
            if col == self.epiconfig.target_column:
                self.column_registration.update_transformation('target', log_params)

            elif col == self.epiconfig.lag_column:
                # only register on lag0; dependents follow via transformation_group
                self.column_registration.update_transformation(
                    f'{self.epiconfig.lag_column}_lag0', log_params
                )

            else:
                self.column_registration.update_transformation(col, log_params)

    def _apply_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log to every column whose TransformationParams has log set."""
        out = df.copy()

        for column_name in self.column_registration.registered_columns:

            col_entry = self.column_registration.get_entry_by_name(column_name)

            if not col_entry.transformation:
                continue

            params = self._resolve_params(col_entry)
            if params is None or params.log is None:
                continue

            out = apply_log(out, col_entry.column_name, params.log)

        return out

    # ── pass 2: compute + register norm params ────────────────────────────

    def _update_columnregistry_norm_params(self, df: pd.DataFrame) -> None:
        """Compute normalisation params from training split and store in registry."""
        if not self.epiconfig.normalization_method:
            return

        method   = self.epiconfig.normalization_method
        train_df = df[df['train']].copy()

        if method not in ('zscore', 'minmax'):
            raise EpiDataOrchestrationError(
                f'Unknown normalization method: {method}. Expected zscore or minmax.'
            )

        # first pass: columns that normalise independently ('self')
        for column_name in self.column_registration.registered_columns:

            col_entry = self.column_registration.get_entry_by_name(column_name)

            if not col_entry.transformation:
                continue
            if col_entry.transformation_group != 'self':
                continue

            if method == 'zscore':
                norm_params = compute_zscore_params(train_df, col_entry.column_name)
            else:
                norm_params = compute_minmax_params(train_df, col_entry.column_name)

            self.column_registration.update_transformation(
                col_entry.column_name, norm_params
            )

        # second pass: columns that follow a reference column
        for column_name in self.column_registration.registered_columns:

            col_entry = self.column_registration.get_entry_by_name(column_name)

            if not col_entry.transformation:
                continue
            if col_entry.transformation_group in (None, 'self'):
                continue

            ref = self.column_registration.get_entry_by_name(col_entry.transformation_group)
            p   = ref.transformation_params

            if p is None:
                continue
            if p.zscore is not None:
                self.column_registration.update_transformation(
                    col_entry.column_name, p.zscore
                )
            elif p.minmax is not None:
                self.column_registration.update_transformation(
                    col_entry.column_name, p.minmax
                )


    # ── pass 3: apply norm ────────────────────────────────────────────────

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored zscore/minmax params to all columns that need it."""
        out = df.copy()

        for column_name in self.column_registration.registered_columns:

            col_entry = self.column_registration.get_entry_by_name(column_name)

            if not col_entry.transformation:
                continue

            params = self._resolve_params(col_entry)
            if params is None:
                continue

            if params.zscore is not None:
                out = apply_zscore(out, col_entry.column_name, params.zscore)
            elif params.minmax is not None:
                out = apply_minmax(out, col_entry.column_name, params.minmax)

        return out

    # ── helper ────────────────────────────────────────────────────────────

    def _resolve_params(self, col_entry: ColEntry) -> TransformationParams | None:
        """
        Return the TransformationParams that govern this column.
        For ``'self'`` columns: the column's own params.
        For referral columns: the reference column's params.
        """
        match (col_entry.transformation, col_entry.transformation_group):

            case (False, _):
                return None

            case (True, 'self'):
                return col_entry.transformation_params

            case (True, str()):
                ref = self.column_registration.get_entry_by_name(
                    col_entry.transformation_group
                )
                return ref.transformation_params

            case (True, None):
                raise ColEntryMissingAttribute(
                    col_entry.column_name, "transformation_params"
                    )                     

            case _:
                assert_never(col_entry.transformation)

    # ── orchestrate ───────────────────────────────────────────────────────

    def orchestrate(self, feature_data: FeatureEpiData) -> TransformedEpiData:
        time_start = time.time()

        df = self._set_splits(feature_data.data.copy())

        # 1. register log params and apply
        self._update_columnregistry_log_params()
        df = self._apply_log(df)

        # 2. compute norm params from training split and register
        self._update_columnregistry_norm_params(df)

        # 3. apply normalisation
        df = self._apply_normalization(df)

        time_end = time.time()

        time_elapsed = time_end - time_start

        return TransformedEpiData(data=df)