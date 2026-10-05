from typing import Literal
import pandas as pd
from abc import abstractmethod

from ..utils.conformal import ResidualScale, ResidualQuantileTable
from ..basemodel import BaseModel
from ...dataloading.databuilders import BaseLineDataBuilder 
from ...dataloading.epidataorchestration.utils.normalization import apply_log, apply_zscore, apply_minmax
from ...utils import crossmark

class BaseLineModel(BaseModel):
    """
    Parent class to all Naive Predictor / BaseLine models.
    This is a first - order sublcass to ``BaseModel``.

    NOTE
    ----
    Baseline models predict from the reverse-transformed (original scale) data in the 
    ``EpiDataOrchestrator``. The predictions are therefore also reverse-transformed. The
    ``PredictionManager`` expects transformed predictions, however, so ``_transform`` 
    transforms the predictions.

    Parameters
    ----------
    databuilder : BaseLineDataBuilder
        Data builder for the model to 'predict' from.
    name : str
        Name of the model.
    residual_scale : {'additive', 'log1p'}
        Scale of the calibration residuals for interval mode. ``'additive'`` keeps
        absolute errors; ``'log1p'`` uses relative errors, which suits nodes of
        different size. See ``models.utils.conformal``.
    min_bin_obs : int
        Seasonal bins with fewer calibration residuals than this fall back to the
        quantiles pooled over all bins.
    calibration_splits : list of {'train', 'val'}
        Splits whose residuals form the calibration pool. The test split is
        never allowed.

    See Also
    --------
    ``BaseModel``
        Parent class to all model classes.
    """
    def __init__(self, 
                 databuilder : BaseLineDataBuilder,                     
                 name : str,
                 residual_scale : ResidualScale = 'additive',
                 min_bin_obs : int = 30,
                 calibration_splits : list[Literal['train', 'val']] | None = None):
        
        self._expected_databuilder = 'BaseLineDataBuilder'

        if calibration_splits is None:
            calibration_splits = ['train', 'val']
        if 'test' in calibration_splits:
            raise ValueError('The test split may never be used for calibration.')

        self.residual_scale     = residual_scale
        self.min_bin_obs        = min_bin_obs
        self.calibration_splits = list(calibration_splits)

        # filled in ``forecast`` when in interval mode; one table per horizon
        self.calibration_tables: dict[int, ResidualQuantileTable] = {}
        
        super().__init__(databuilder,  name)

        self.config_info.update({'residual_scale'    : residual_scale,
                                 'min_bin_obs'       : min_bin_obs,
                                 'calibration_splits': self.calibration_splits})

        # Following statuses are not applicable
        self.status_dict.pop('model_hparams_set')
        self.status_dict.pop('global_hparams_set')
        self.status_dict.pop('trained') 

    @abstractmethod
    def forecast(self, dataset: Literal['train','val','test'] = 'test') -> None:
        pass

    # ====== NONSENSE METHODS ====== #
    # methods that are not actually used in naive predictors
    def train(self, *args, **kwargs) -> None:
        print("This BaseLineModel doesn't train")

    def set_global_hparams(self, *args, **kwargs) -> None:
        print("This BaseLineModel doesn't have global hyper parameters")

    def set_model_hparams(self, *args, **kwargs) -> None:
        print("This BaseLineModel doesn't have model hyper parameters") 

    def save_model(self, *args, **kwargs) -> None:
        print(f'{crossmark} Baseline models cant be saved.')
    
    def calibration_summary(self) -> pd.DataFrame:
        """
        Number of calibration residuals per seasonal bin and horizon, and whether the
        bin fell back to the pooled quantiles. Use this to judge whether the
        week-of-year grouping is too thin.
        """
        if not self.calibration_tables:
            raise ValueError('No calibration tables: run forecast() in interval mode first.')

        frames = []
        for hh, tbl in self.calibration_tables.items():
            df = tbl.n_obs.reset_index()
            df.columns = ['seasonal_index', 'n_obs']
            df['horizon']  = hh
            df['fallback'] = df['n_obs'] < tbl.min_obs
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # ======= HIDDEN METHODS ======= 
    def _get_seasonal_index(self, df: pd.DataFrame) -> pd.Series:
        """Seasonal index (week of year, day of year or month) based on temporal frequency."""
        freq = self.databuilder.dataorchestrator.config.temporal_frequency
        if freq == 'w':
            return df[self.epiconfig.temporal_column].dt.isocalendar().week.astype(int)
        elif freq == 'd':
            return df[self.epiconfig.temporal_column].dt.dayofyear.astype(int)
        elif freq == 'm':
            return df[self.epiconfig.temporal_column].dt.month.astype(int)
        else:
            raise ValueError(f'Invalid temporal frequency for {self.model_class}: {freq}')

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to baseline model predictions, bringing them into the 
        transformed scale expected by PredictionManager.
        """
        if self.databuilder.dataorchestrator.config.target_column != 'incidence':
            return df.copy()

        col_entry = self.column_registration.get_entry_by_name('target')
        params    = col_entry.transformation_params

        if params is None:
            return df.copy()

        columns       = ['target'] + self.prediction_columns
        df_transformed = df.copy()

        for col in columns:

            if col not in df_transformed.columns:
                continue

            if params.log is not None:
                df_transformed = apply_log(df_transformed, col, params.log)

            if params.zscore is not None:
                df_transformed = apply_zscore(df_transformed, col, params.zscore)

            elif params.minmax is not None:
                df_transformed = apply_minmax(df_transformed, col, params.minmax)

        return df_transformed