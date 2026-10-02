from typing import Literal
import pandas as pd
from abc import abstractmethod

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

    See Also
    --------
    ``BaseModel``
        Parent class to all model classes.
    """
    def __init__(self, 
                 databuilder : BaseLineDataBuilder,                     
                 name : str,):
        
        self._expected_databuilder = 'BaseLineDataBuilder'
        
        super().__init__(databuilder,  name)

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
    
    # ======= HIDDEN METHODS ======= 
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

        columns       = ['target'] + [self.pred_col]
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