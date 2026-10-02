from ...dataloading.databuilders import BaseLineDataBuilder 
from .baselinemodel import BaseLineModel 

from ...utils import DataSetSplit

class Persistence(BaseLineModel):
    """ 
    Persistence model returns the most recent observation as prediction.

    See Also
    --------
    ``BaseModel``
        Parent class of all models.
    ``BaseLineModel``
        Parent class of baseline models.
    """
    def __init__(self, 
                 databuilder : BaseLineDataBuilder,                 
                 name: str = 'persistence_model'):
        
        super().__init__(databuilder, name)

    def forecast(self, dataset: DataSetSplit = 'test') -> None:
        """
        Forecast for set dataset
        """
        assert isinstance(self.databuilder, BaseLineDataBuilder)

        for hh in range(self.databuilder.dataorchestrator.config.horizon_size):

            # get shift between target and pred
            timeshift_num = int(hh + self.databuilder.dataorchestrator.config.horizon_leadtime)
            evaluation_df = self.databuilder.dataloader_main

            evaluation_df = evaluation_df.sort_values([
                self.epiconfig.id_column, 
                self.epiconfig.temporal_column]
            ).copy()

            # group by, and shift 'target' by ``timeshift_num``.
            # that is the prediction: the shifted 'target'.
            persistence_pred = evaluation_df.groupby(
                self.epiconfig.id_column
                )['target'].shift(timeshift_num)

            evaluation_df[self.pred_col] = persistence_pred

            # filter on dataset train/val/test
            evaluation_df = evaluation_df[evaluation_df[dataset]]
            evaluation_dataset = evaluation_df[
                [self.epiconfig.id_column, self.epiconfig.temporal_column, 'target'] + [self.pred_col]
                ]

            self.predictions.add_horizon_predictions(dataset, self._transform(evaluation_dataset), hh)

        self._update_status('forecasted')   