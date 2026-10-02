from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any
from pathlib import Path
import yaml
import dataclasses

from .pathmanager import EpiPathsManager
from .validator import EpiConfigValidator
from .exceptions import EpiConfigValidationError, IncompatibleEpiConfigs

from ...utils import align, return_header_line, Country, AdminLevel, Disease, InvalidExtension

import logging
logger = logging.getLogger(__name__)

@dataclass
class EpiConfig:
    """
    Configuration dataclass that dictates how ``EpiDataOrchestrator`` turns raw data
    into model-ready dataloaders.

    Parameters
    ----------
    disease : Disease   
    temporal_frequency : Literal['m','w','d']= 'w'
    min_date : str = '2011-01-01'
    max_date : str = '2020-06-01'
    split_trainval : str = '2018-06-01'
    split_valtest : str = '2019-06-01'
 
    country: Country = 'germany'
    level: AdminLevel = 'nuts3'
 
    horizon_size : int = 1
    horizon_leadtime : int = 1

    time_index_d : bool = False
    time_index_w : bool = True
    time_index_m:  bool = False
    lag_column : str  = 'incidence'
    lag_num : int  = 1
    sequence_length : int  = 1
    incidence_scalar : int  = 10_000    

    feature_popsize : bool = False      
    feature_popdens : bool = False

    normalization_method : Literal['minmax', 'zscore'] | None = 'zscore'
    log_transform : list[str] | None = None
    log_shift : float = 1.0        
            
    temporal_column : str = 'timestamp'
    target_column : str = 'incidence'
    id_column : str = 'node'
    pred_column : str = 'pred'

    Methods
    -------
    ``save_config()``
        Save an EpiConfig to a path (`.yaml`).
    ``load_config()``
        Load an EpiConfig from a path (`.yaml`).
    ``copy()``
        Copy an EpiConfig.
    ``get_summary()``
        Get a visually pleasing an understandable summary of ``EpiConfig``.

    See Also
    --------
    ``EpiDataOrchestrator``
        Uses an ``Epiconfig`` to create a model-ready dataframe.

    ``EpiPathsManager``
        Helper class to ``EpiConfig`` that stores paths.

    ``EpiConfigValidator``
        Helper class to ``EpiConfig`` that validates the input and paths.

    Downstream
    ----------
    ``EpiConfig`` stores all configuration information needed to transform raw data
    into model-ready datasets by ``EpiDataOrchestrator``. ``EpiValidator`` validates the
    input to ``EpiConfig``, as well as the paths that ``EpiPathsManager`` stores.  

    Any model instance uses a ``BaseLineDataBuilder`` or a ``GraphDataBuilder``. These 
    are both created based on the same ``FinalizedEpiData`` instance of the 
    ``EpiDataOrchestrator``.       

    Examples
    --------
    >>> experiment_1a_epicfg = EpiConfig(
    ...     disease             = 'influenza',
    ...     temporal_frequency  = 'w',
    ...     min_date            = '2012-06-01',
    ...     max_date            = '2020-06-01',
    ...     split_trainval      = '2018-06-01',
    ...     split_valtest       = '2019-06-01' ,  
    ...     country             = 'germany',
    ...     level               = 'nuts3',
    ...     horizon_size        = 1,
    ...     horizon_leadtime    = 1,
    ...     time_index_d        = False,
    ...     time_index_m        = False,
    ...     time_index_w        = True,
    ...     lag_column          = 'incidence',
    ...     lag_num             = 1, 
    ...     sequence_length     = 4,
    ...     incidence_scalar    = 10_000,
    ...     feature_popdens     = True,
    ...     feature_popsize     = True,
    ...     normalization_method= 'zscore',
    ...     log_transform       = ['incidence'],
    ...     log_shift           = 1,
    ...     temporal_column     = 'timestamp',
    ...     target_column       = 'incidence',
    ...     id_column           = 'node',
    ...     pred_column         = 'pred'
    ...     )
    >>> experiment_1a_epicfg = EpiConfig.load_config(
    ...     self.path_exp / self.epicfg_filename
    ...     )     
    ... edo = EpiDataOrchestrator(experiment_1a_epicfg).build()    

    """
    # ============= MAIN =============
    disease : Disease   
    
    # ============= TEMPORAL =============
    temporal_frequency : Literal['m','w']= 'w'
    min_date : str = '2011-01-01'
    max_date : str = '2020-06-01'
    split_trainval : str = '2018-06-01'
    split_valtest : str = '2019-06-01'
    
    # ============= GEOGRAPHY =============
    country : Country = 'germany'
    level : AdminLevel = 'nuts3'
    
    # ============= TASK =============
    horizon_size : int = 1
    horizon_leadtime : int = 1
    
    # ============= FEATURES =============
    time_index_d : bool = False
    time_index_w : bool = True
    time_index_m : bool = False
    lag_column : str  = 'incidence'
    lag_num : int  = 1
    sequence_length : int  = 1
    incidence_scalar : int  = 10_000    

    feature_popsize : bool = False      
    feature_popdens : bool = False

    # ============= NORMALIZATION =============
    normalization_method : Literal['minmax', 'zscore'] | None = 'zscore'
    log_transform : list[str] | None = None
    log_shift : float = 1.0        
            
    # ============= COLUMN NAMES =============
    temporal_column : str = 'timestamp'
    target_column : str = 'incidence'
    id_column : str = 'node'
    pred_column : str = 'pred'

    # ============= DUNDER ============ #
    def __post_init__(self):
        
        # set pathmanager
        self.path_manager   = EpiPathsManager(self.country, self.level, self.disease)
        
        self.validator      = EpiConfigValidator(self)
        self.validator.validate()

        self._set_hidden_attributes()
        self._classify_attributes()

        logger.debug('EpiConfig has been created')           

    # ============ Methods =========== #
    def assert_equals(self, 
                      other: EpiConfig | dict[str,str], 
                      level: Literal[0,1,2,3,4] = 1) -> None:
        """for DeepModel - loading use level = 1. For Evaluator use level = 2!"""        
        self_summary  = self.get_summary(level)
        if isinstance(other, dict):
            other_summary = other
        else:
            other_summary = other.get_summary(level)
        diff = {k: (self_summary[k], other_summary.get(k))
                for k in self_summary if self_summary[k] != other_summary.get(k)}
        if diff:
            raise IncompatibleEpiConfigs(f"EpiConfig mismatch at level {level}:\n" + 
                            "\n".join(f"  {k}: {v[0]} vs {v[1]}" for k, v in diff.items()))

    # ============ CONFIG LOADING/SAVING ==============
    def save_config(self, path: Path):
        """ 
        saves EpiConfig to a .yaml
        """
        if path.suffix != '.yaml':
            raise InvalidExtension('.yaml',path.suffix)
        
        config_dict = dataclasses.asdict(self)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)    

    def copy(self, overrides : dict[Any, Any]) -> EpiConfig:
        """
        Returns a new EpiConfig instance with the same settings.
        Optionally override specific fields by passing them as keyword arguments.

        Example:
            new_cfg = cfg.copy(max_date='2021-01-01', horizon_size=2)
        """
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        fields.update(**overrides)
        return EpiConfig(**fields)

    @classmethod
    def load_config(cls, path: Path) -> EpiConfig:
        """ 
        Loads a .yaml of name `config_name` into an EpiConfig
        the directory returned by `get_config_path()`.
        """        
        if not str(path).endswith('.yaml'):
            raise ValueError('path must end with .yaml')

        with open(path) as f:
            d = yaml.safe_load(f)        
        return cls(**d)      

    # ============= ATTRIBUTE ORGANIZATION ==============
    def _set_hidden_attributes(self) -> None:
        """post validation, sets hidden attributes that should be accessed (and are not available in representation)"""
        self._num_quantiles = 0

    def _classify_attributes(self) -> None:
        """creates dictionaries of attributes and classifies those. Used for back-end and for interaction with repr/str dunders"""

        self.attributes_dict = vars(self)

        self.attributes_classified_dict = {
            'main'          :   ['disease'],
            'temporal'      :   ['temporal_frequency','min_date','max_date','split_trainval','split_valtest'],
            'geography'     :   ['country','level'],
            'task'          :   ['horizon_size','horizon_leadtime'],
            'features'      :   ['time_index_d','time_index_w','time_index_m','lag_column','lag_num','sequence_length','incidence_scalar', 'feature_popsize','feature_popdens'],
            'normalization' :   ['normalization_method','log_transform','log_shift'],    
            'column_names'  :   ['temporal_column','target_column','id_column','pred_column'],
            'none'          :   ['attributes_dict', 'attributes_classified_dict'],
            'helper_classes':   ['path_manager','validator']
        }

        for attribute in self.attributes_dict:
            # hidden attributes are not of interest
            if attribute.startswith("_"):
                continue

            classified = any(attribute in value_list for value_list in self.attributes_classified_dict.values())
            if not classified:
                raise EpiConfigValidationError(f'Attribute {attribute} not classified.\nLikely stems from an update in EpiConfig class, without incorporating it into the classification dict in _classify_attributes()')

    # ============= PROPERTIES =============
    @property
    def split_columns(self) -> list[str]:
        """Names of split indicator columns."""
        return ['train', 'val', 'test']
    
    # ============= DICT - SUMMARIES =========== #
    def get_summary(self, level: Literal[0, 1, 2, 3, 4]) -> dict[str, str]:
        """
        Level 0: core identity only
        Level 1: + temporal, excl. max_date  (use for model loading — test period may differ)
        Level 2: + temporal incl. max_date   (use for evaluation — test period must match)
        Level 3: + features, normalization
        Level 4: all attributes        
        """
        CLASSES_BY_LEVEL = {
            0: {'main', 'geography', 'task', 'column_names'},
            1: {'main', 'geography', 'task', 'column_names', 'temporal'},
            2: {'main', 'geography', 'task', 'column_names', 'temporal'},            
            3: {'main', 'geography', 'task', 'column_names', 'temporal', 'features', 'normalization'},
            4: None,  # None = all classes
        }
        EXCLUDE_BY_CLASS = {
            'temporal': {'max_date'},  # testing period may differ
        }

        allowed_classes = CLASSES_BY_LEVEL[level]
        summary: dict[str, str] = {}

        for attr_class, attr_list in self.attributes_classified_dict.items():
            if allowed_classes is not None and attr_class not in allowed_classes:
                continue

            # only exclude at level 1; at level 2+ max_date is included
            exclude = EXCLUDE_BY_CLASS.get(attr_class, set())
            
            if level > 1:
                exclude = set()

            for attr_name, attr_value in self.attributes_dict.items():
                if attr_name in attr_list and attr_name not in exclude:
                    summary[attr_name] = str(attr_value)

        return summary

    def __repr__(self) -> str:
        repr =(
            f"<{self.__class__.__name__}(disease={self.disease}, "
                f"country={self.country}, "                
                f"level={self.level}, "
                f"min_date={self.min_date}, "
                f"max_date={self.max_date}, "          
                f"horizon_leadtime={self.horizon_leadtime}, "                      
                f"horizon_size={self.horizon_size}, "
                f"sequence_length={self.sequence_length})"  
        )          
        return repr
    
    def __str__(self) -> str:
        """extensive - summary: all attributes are displayed"""
        all_keys        = list(self.attributes_dict.keys())
        width           = max(len(k) for k in all_keys)
        indent          = 4
        
        lines = [f"<{self.__class__.__name__}("]     

        for attr_class, attr_list in self.attributes_classified_dict.items():
            if attr_class != 'none':
                lines.append(return_header_line(attr_class, n_indent_chars=12, indent = indent))
                for attr_name, attr_value in self.attributes_dict.items():
                    if attr_name in attr_list:
                        lines.append(align(attr_name, attr_value, width, indent = indent))
                lines.append("")

        lines.append(")>")
        repr = '\n'.join(lines)        
        return repr            