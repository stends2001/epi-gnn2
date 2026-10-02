import pandas as pd 
from dataclasses import dataclass

from ....utils.textformatting import checkmark


@dataclass
class FeatureEpiData:
    """
    Data container during data-orchestration for stage 4: feature data.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame with features as requested in ``EpiConfig``. At least the following
        columns are expected:
        - ``{epiconfig.temporal_column}``
        - ``{epiconfig.id_column}``
        - ``{epiconfig.target_column}``

        with the rest depending on ``EpiConfig``.

    See Also
    --------
    ``EpiFeatureBuilder``
        Loads context data into ``FeatureEpiData``.
    ``FeatureValidator``        
        Validates the integrity of ``FeatureEpiData``.          

    Downstream
    ----------
    ``EpiDataOrchestrator`` loads and processes the raw data, and stores it in
    intermediate data containers. ``FeatureEpiData`` is the fourth of these.
    """   
    data : pd.DataFrame

    def __repr__(self):
        representation = (f"<{self.__class__.__name__}(data {checkmark}")
        representation += ")>"
        return representation  