from dataclasses import dataclass

@dataclass
class ZScoreParams:
    """
    Simple dataclass that stores zscore - normalization parameters
    
    See Also
    --------
    ``apply_zscore``
        Executes the zscore transform.
    """
    mean: float
    std:  float

@dataclass
class MinMaxParams:
    """
    Simple dataclass that stores minmax - normalization parameters
    
    See Also
    --------
    ``apply_minmax``
        Executes the minmax transform.    
    """    
    min: float
    max: float

@dataclass
class LogParams:
    """
    Simple dataclass that stores log - transformation parameters
    
    See Also
    --------
    ``apply_log``
        Executes the log transform.
    """    
    shift: float = 1.0

@dataclass
class TransformationParams:
    """
    Holds all transformation parameters for a single ``ColEntry``.
    
    Parameters
    ----------
    log : LogParams | None = None
        Parameters dictating the log-transform applied to this column.
    zscore : ZScoreParams | None = None
        Parameters dictating the zscore-transform applied to this column.
    minmax: MinMaxParams | None = None
        Parameters dictating the minmax-transform aplied to this column.

    Attributes
    ----------
    ``has_transformation``
        Whether or not a transformation (log) is saved.
    ``has_normalization``
        Whether or not a normalization (zscore/minmax) is saved    

    See Also
    --------
    ``ColEntry``
        Column metadata, that may store ``TransformationParams``.

    DownStream
    ----------
    ``EpiDataOrchestrator`` Stores information on columns in ``ColEntry`` instances,
    which are stored in ``ColumnRegistry``. Using ``TransformationParams``, we can
    keep track, and later reverse, transformations executed.
    """
    log : LogParams | None = None
    zscore : ZScoreParams | None  = None
    minmax : MinMaxParams | None  = None

    @property
    def has_transformation(self) -> bool:
        return self.log is not None

    @property  
    def has_normalization(self) -> bool:
        return self.zscore is not None or self.minmax is not None