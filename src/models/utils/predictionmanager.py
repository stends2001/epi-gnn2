import pandas as pd 
from typing import Literal
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from .predictioncollection import PredictionCollection

from .exceptions import InvalidPredictionsError
from ...dataloading.epidataorchestration.utils.normalization import reverse_log, reverse_minmax, reverse_zscore
from ...dataloading.epidataorchestration.orchestrator import EpiDataOrchestrator
from ...dataloading.columnregistration import ColumnRegistry
from ...dataloading.epidataorchestration.utils.temporal_summary import EpiDataTemporalSummary, TemporalError
from ...utils import DataSetSplit


class PredictionManager:
    """
    Manages a single model's predictions in a centralized class.

    Within this class, each of ['train','val','test'] is an instance of 
    ``PredictionCollection``.

    NOTE
    ----
    The predictions and the targets are shifted here back to how they'd be
    observed in reality. Thus, with respect to the dataorchestrator's 
    final data, the timestamp/target is shifted.
    
    Parameters
    ----------
    epiconfig : EpiConfig
        Large configuration class that dictates which data to load.    
    column_registration : ColumnRegistry
        Registry of columns that keeps track of all columns and transformations.
    temporal_summary : EpiDataTemporalSummary
        Helper class that stores temporal information, built based on ``EpiConfig``.

    Methods
    -------
    ``add_horizon_predictions()``
        Adds data to ``PredictionManager``.
    ``get_preds()``
        Gets data from ``PredictionManager``.

    There are additional hidden (internal) functions that supper the ones listed above.

    See Also
    --------
    ``PredictionCollection``
        Stores predictions across horizons for a single datast (train/val/test)    

    Downstream
    ----------
    Each mode class is a subclass of ``BaseModel``, which creates an instance of 
    ``PredictionManager``. These in turn consist of three ``PredictionCollection``s.    
    """

    def __init__(self, 
                 data_orchestrator : EpiDataOrchestrator, 
                 column_registration : ColumnRegistry,
                 temporal_summary : EpiDataTemporalSummary):

        self.train = PredictionCollection()
        self.val   = PredictionCollection()
        self.test  = PredictionCollection()
        
        self.epidata_orchestrator   = data_orchestrator
        self.epiconfig              = data_orchestrator.config
        self.column_registration    = column_registration
        self.temporal_summary       = temporal_summary

        # call helper functions
        self._setup_required_columns()

    def add_horizon_predictions(self, 
                                dataset : Literal['train','val','test'], 
                                horizon_df : pd.DataFrame, 
                                horizon : int):
        """
        Adds a prediction df to the prediction manager

        Internally validates columns, temporal-axis, and everything else.
        Internally adds transformed and non-transformed data, as well as
        spatially-aggregated data.

        Parameters
        ----------
        dataset : Literal['train','val','test']
            Which dataset (thus which ``PredictionCollection``) the data belongs in.
        horizon_df : pd.DataFrame
            The actual data for these parameters. Columns are validated.
        horizon : int 
            The horizon to store. Starts at zero, irrespective of horizon lead time.
        """
 
        df_validated    = self._validate_columns(horizon_df)

        # shift timestamps to match the PREDICTION timestamps
        df_shifted_t    = self._shift_prediction_timestamp(df_validated)
        df_filtered     = self._filter_by_dataset_timerange(df_shifted_t, dataset)
        
        self._validate_predictions_temporally(df_filtered, dataset)
                
        # get prediction_collection
        prediction_collection = self._return_dataset(dataset)

        # Add both normalized and denormalized versions, as well as a national aggregate for the denormalized
        df_transformed      = df_filtered.copy()
        df_nontransformed   = self._denorm_predictions(df_transformed.copy())
        df_aggregated       = self._aggregate_predictions_spatially(df_nontransformed.copy())
        
        # transformed - regional
        prediction_collection.add(df_transformed, 
                                  horizon               = horizon, 
                                  is_original           = False, 
                                  spatially_aggregated  = False)
        # non-transformed - regional
        prediction_collection.add(df_nontransformed, 
                                  horizon               = horizon, 
                                  is_original           = True, 
                                  spatially_aggregated  = False)
        # non-transformed - aggregated
        prediction_collection.add(df_aggregated,
                                  horizon               = horizon, 
                                  is_original           = True, 
                                  spatially_aggregated  = True)

    def get_preds(self, dataset: DataSetSplit) -> 'PredictionCollection':
        """
        get predictions from any of the datasets
        """
        return self._return_dataset(dataset)

    def _return_dataset(self, dataset: DataSetSplit) -> 'PredictionCollection':
        """
        Returns the PredictionCollection at attribute train/val/test
        if invalid dataset, error is raised from decorator
        """
        if dataset == 'train':
            return self.train
        elif dataset == 'val':
            return self.val 
        elif dataset == 'test':
            return self.test 

    def _setup_required_columns(self):
        """
        sets the required columns in the model-specific prediction dfs.
        The columns present should be:
        - temporal_column (epiconfig.temporal_column)
        - node_column (epiconfig.id_column)
        - 'target'
        - prediction columns:
            - if num_qunatiles = 0 => only 'pred'
            - else all registered pred columns except for 'pred'
        """        
        cols        = [self.epiconfig.temporal_column, self.epiconfig.id_column, 'target']

        if self.epiconfig._num_quantiles == 0:
            pred_cols= ['pred']
        else:
            pred_cols= [c for c in self.column_registration.pred_columns if c != 'pred']

        self.pred_cols          = pred_cols
        self.required_columns   = cols + pred_cols

    def _shift_prediction_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        In the DataOrchestrator the target has been shifted. 
        In other words, the combination of values for columns
        | 'timestamp' | 'target' | has a temporal discrepancy,
        where 'timestamp' refers to t0; the point in time of the
        most recent observation (related to features, not target).

        In order to plot and analyze correctly, the prediction
        managers deal with the 'correct' temporal axis: target/pred
        are looked at from the correct point in time. Thus, we shift
        the target by {epiconfig.horizon_leadtime} timesteps, of the 
        proper temporal_frequency.
        """
        dfc                                 = df.copy()

        if self.epiconfig.temporal_frequency == 'd':
            delta = timedelta(days=self.epiconfig.horizon_leadtime)
        elif self.epiconfig.temporal_frequency == 'w':
            delta = timedelta(weeks=self.epiconfig.horizon_leadtime)
        elif self.epiconfig.temporal_frequency == 'm':
            delta = relativedelta(months=self.epiconfig.horizon_leadtime)
        else:
            raise TemporalError(f"Unknown frequency: {self.epiconfig.temporal_frequency}")

        timestamps: pd.Series[pd.Timestamp] = dfc[self.epiconfig.temporal_column]
        dfc[self.epiconfig.temporal_column]  = timestamps + delta

        return dfc

    def _denorm_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalizes the normalized predictions in df using the parameters
        from the column registry.
        """
        df_denorm = df.copy()

        if self.epiconfig.normalization_method:
            col_entry = self.column_registration.get_entry_by_name('target')

            if col_entry.transformation and col_entry.transformation_params is not None:
                params = col_entry.transformation_params

                for col in self.column_registration.pred_columns + ['target']:
                    if col not in df_denorm.columns:
                        continue

                    if params.zscore is not None:
                        df_denorm = reverse_zscore(df_denorm, col, params.zscore)
                    elif params.minmax is not None:
                        df_denorm = reverse_minmax(df_denorm, col, params.minmax)

                    if params.log is not None:
                        df_denorm = reverse_log(df_denorm, col, params.log)

        if self.epiconfig.target_column == 'cases':
            raise ValueError('not dealt with poissonlosses')

        return df_denorm

    def _validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the df with all the required columns
        Also validates that these are all present
        """
        essential_columns = self.required_columns
        
        for cc in essential_columns:
            if cc not in data.columns.tolist():
                raise InvalidPredictionsError(f'column {cc} not found in prediction df')
            
        for cc in data.columns:
            if cc not in essential_columns:
                raise InvalidPredictionsError(f'column {cc} found in prediction df but not expected')
                        
        return data[self.required_columns].copy()  
    
    def _get_anchor_for_merge(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetches the anchor column from data_final.data and merges it onto df
        by timestamp and node id, so _denorm_predictions can use it for delta reversal.
        """
        anchor_col = f'{self.epiconfig.target_column}_anchor'
        merge_keys = [self.epiconfig.temporal_column, self.epiconfig.id_column]

        source_df = self.epidata_orchestrator.data_final.data[merge_keys + [anchor_col]]
        
        return df.merge(source_df, on=merge_keys, how='left')

    def _filter_by_dataset_timerange(self, df: pd.DataFrame, dataset: DataSetSplit) -> pd.DataFrame:
        """
        Filters out data from outside the expected period,
        using the dates specified in TemporalSummary
        """
        if dataset == 'train':
            min_date = self.temporal_summary.min_date
            max_date = self.temporal_summary.split_trainval
        elif dataset == 'val':
            min_date = self.temporal_summary.split_trainval
            max_date = self.temporal_summary.split_valtest
        elif dataset == 'test':
            min_date = self.temporal_summary.split_valtest
            max_date = self.temporal_summary.max_date_extended

        dfc     = df.copy()
        mask    = (dfc[self.epiconfig.temporal_column] >= min_date) & (dfc[self.epiconfig.temporal_column] < max_date)
        filtered= dfc[mask].reset_index(drop=True)        
        return filtered

    def _validate_predictions_temporally(self, df: pd.DataFrame, dataset: DataSetSplit):
        dt_index = pd.DatetimeIndex(
            pd.to_datetime(df[self.epiconfig.temporal_column])
        ).drop_duplicates().sort_values()

        expected_timerange = self.temporal_summary.get_daterange_dataset(
            dataset,
            reference='target'
        )

        # ── single-day test sets: just check the date is correct ─────────────
        if len(dt_index) < 3:
            expected_start = pd.Timestamp(expected_timerange[0])
            expected_end   = pd.Timestamp(expected_timerange[1])
            actual_dates   = set(dt_index)
            expected_dates = set(pd.date_range(start=expected_start, end=expected_end))

            missing  = expected_dates - actual_dates
            leftover = actual_dates - expected_dates

            if missing:
                raise InvalidPredictionsError(
                    f"Prediction timestamps incomplete. Missing: {sorted(missing)}"
                )
            if leftover:
                raise InvalidPredictionsError(
                    f"Prediction timestamps too numerous. Leftover: {sorted(leftover)}"
                )
            return

        # ── normal case: infer frequency and validate range ───────────────────
        freq = pd.infer_freq(dt_index)
        if freq is None:
            raise InvalidPredictionsError(
                "Could not infer frequency from prediction timestamps."
            )

        range_expected = pd.date_range(
            start=expected_timerange[0],
            end=expected_timerange[1],
            freq=freq
        )

        missing = range_expected.difference(dt_index)
        if not missing.empty:
            raise InvalidPredictionsError(
                f"Prediction timestamps are incomplete. Missing: {missing.tolist()}"
            )

        leftover = dt_index.difference(range_expected)
        if not leftover.empty:
            raise InvalidPredictionsError(
                f"Prediction timestamps are too numerous. Leftover: {leftover.tolist()}"
            )
        
    def _aggregate_predictions_spatially(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates regional predictions to a national level.
        
        For 'cases': simple sum across regions per timestep.
        For 'incidence': population-weighted aggregation —
            converts to cases, sums nationally, divides by total population.
        """
        dfc = df.copy()
        columns_to_aggregate = self.pred_cols + ['target']
        temporal_col = self.epiconfig.temporal_column

        if self.epiconfig.target_column == 'cases':
            aggregated = (
                dfc
                .groupby(temporal_col)[columns_to_aggregate]
                .sum()
                .reset_index()
            )

        elif self.epiconfig.target_column == 'incidence':
            population_data = self.epidata_orchestrator.data_context.population_size

            dfc['year'] = dfc[temporal_col].dt.year
            dfc = dfc.merge(population_data, on=[self.epiconfig.id_column, 'year'])

            # Convert incidence rates -> raw cases per region
            rate_denominator = self.epiconfig.incidence_scalar
            for col in columns_to_aggregate:
                dfc[f'{col}_cases'] = dfc[col] * dfc['population_size'] / rate_denominator

            # Sum cases and population nationally per timestep
            case_cols = [f'{col}_cases' for col in columns_to_aggregate]
            national = (
                dfc
                .groupby(temporal_col)[case_cols + ['population_size']]
                .sum()
                .reset_index()
            )

            # Convert back to national incidence rate
            for col in columns_to_aggregate:
                national[col] = national[f'{col}_cases'] / national['population_size'] * rate_denominator

            aggregated = national[[temporal_col] + columns_to_aggregate]

        else:
            raise ValueError(f"Unsupported target_column '{self.epiconfig.target_column}' for spatial aggregation.")

        return aggregated

    def __repr__(self) -> str:
        status_return = ""

        if self.train._contains_data():
            status_return += "train, "
        if self.val._contains_data():
            status_return += "val, "
        if self.test._contains_data():
            status_return += "test, "  

        if len(status_return) == 0:
            status_return += "No predictions found"
        else:
            # remove final space and comma 
            status_return = f"PredictionCollection at {status_return[:-2]}"                     

        return f"<PredictionManager({status_return})>"
