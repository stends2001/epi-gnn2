import pandas as pd 
from dataclasses import dataclass

from ....utils.textformatting import checkmark

@dataclass
class TransformedEpiData:
    """
    Data container during data-orchestration for stage 5: transformed data.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed data frame with features as requested in ``EpiConfig``. At least 
        the following columns are expected:
        - ``{epiconfig.temporal_column}``
        - ``{epiconfig.id_column}``
        - ``{epiconfig.target_column}``
        - ``'train'``
        - ``'val'``
        - ``'test'``        

        with the rest depending on ``EpiConfig``.

    See Also
    --------
    ``EpiDataTransformer``
        Loads context data into ``TransformedEpiData``.
    ``TransformedValidator``        
        Validates the integrity of ``TransformedEpiData``.                         

    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. ``TransformedEpiData`` is the fifth of these.
    """   
    data : pd.DataFrame  

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
        representation += ")>"
        return representation