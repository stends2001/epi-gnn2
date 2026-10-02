import pandas as pd 
from dataclasses import dataclass

from ....utils.textformatting import checkmark

@dataclass
class FinalizedEpiData:
    """
    Data container during data-orchestration for stage 6: finalized data.

    Parameters
    ----------
    data : pd.DataFrame
        Final transformed data. This dataset will be the foundation of 
        ``GraphDataBuilder``. The entire set of columns depends on the features
        specified in ``EpiConfig``, but at least the following columns are expected:
        - ``{epiconfig.temporal_column}``
        - ``{epiconfig.id_column}``
        - ``{epiconfig.target_column}_lead{H}``
        - ``'train'``
        - ``'val'``
        - ``'test'``        

        where H is the integer representing the minimal horizon_leadtime as defined as
        in ``EpiConfig``.
    data_denorm : pd.DataFrame
        Final reverse-transformed (original scale) data. This dataset will be the 
        foundation of ``BaselineDataBuilder``. The entire set of columns depends on the
        features specified in ``EpiConfig``, but at least the following columns are 
        expected:
        - ``{epiconfig.temporal_column}``
        - ``{epiconfig.id_column}``
        - ``{epiconfig.target_column}_lead{H}``
        - ``'train'``
        - ``'val'``
        - ``'test'``  

        where H is the integer representing the minimal horizon_leadtime as defined as
        in ``EpiConfig``.             

    See Also
    --------
    ``EpiDataFinalizer``
        Loads context data into ``FinalizedEpiData``.
    ``FinalizedValidator``        
        Validates the integrity of ``FinalizedEpiData``.                 

    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. ``FinalizedEpiData`` is the sixth (and last) of these.
    """    
    data : pd.DataFrame
    data_denorm : pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark},"
                f"data_denorm {checkmark}"
                )
        representation += ")>"
        return representation   