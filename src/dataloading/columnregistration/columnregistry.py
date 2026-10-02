from dataclasses import dataclass, field

from .colentry import ColEntry
from .exceptions import MissingColEntry, MissingTransformationReferral, TransformationParamsAlreadySet 
from .transformation_params import TransformationParams, LogParams, ZScoreParams, MinMaxParams

from ...utils.textformatting import align
from ...utils.types import ColumnType

import logging
logger = logging.getLogger(__name__)

@dataclass
class ColumnRegistry:
    """
    Stores and manages column metadata through ``ColEntry`` instances. These are used in
    the data-preparation pipeline guided through ``EpiDataOrchestrator``. Provides easy 
    access to columns by type and normalization information.

    Takes no parameters

    Methods
    -------
    ``add_column()``   
        Add a ``ColumnEntry`` .
    ``update_transformation()``
        Update transformation parameters associated with a ``ColumnEntry``.

    ``get_entries_names_by_type()``
        Get a list of the names of all saved ``ColumnEntry``s.
    ``get_entries_by_type()``
        Get a list of the ``ColumnEntry``s by type.
    ``get_transformation_groups()``
        Get a list of all transformation groups of the saved ``ColumnEntry``s.
    ``get_entry_by_name()``
        Get the ``ColumnEntry`` associated with a name.

    Attributes
    ----------
    ``context_columns``
        List of column names with ``column_type`` == ``'context'``.
    ``feature_columns``
        List of column names with ``column_type`` == ``'feature'``.    
    ``target_columns``
        List of column names with ``column_type`` == ``'target'``.    
    ``pred_columns``
        List of column names with ``column_type`` == ``'pred'``.    
    ``split_columns``
        List of column names with ``column_type`` == ``'split'``.    
    ``registered_column``
        List of all column names.    

    See Also
    --------
    ``ColEntry`` 
        A single entry stored in ``ColumnRegistry`` that contains metadata
        of a single column.

    Downstream
    ----------
    ``EpiDataOrchestrator`` Stores information on columns in ``ColEntry`` instances,
    which are stored in ``ColumnRegistry``.

    Examples
    --------
    >>> column_registration = ColumnRegistry()
    >>> column_registration.add_column(
    ...     'target', 
    ...     'target',
    ...     transformation = True,
    ...     transformation_group = 'self'
    ...     )           
    >>> log_params = LogParams(shift=self.epiconfig.log_shift)
    ... column_registration.update_transformation(
    ...     'target', 
    ...     log_params
    ...     )
    """
    _entries: list[ColEntry] = field(default_factory=list[ColEntry])
    
    # ========= ADJUSTING / UPDATING COLUMNREGISTRATION ======= #
    def add_column(self, 
                   column_name : str, 
                   column_type : ColumnType, 
                   transformation : bool = False,                   
                   transformation_group : str | None = None, 
                   transformation_params : TransformationParams | None  = None):
        """
        Adds columns to its list of entries. All inputs here correspond to those of
        ``ColEntry``. For further information, please see ``ColEntry``.
        """
        # if transformation is guided by another column, validate that that column 
        # already exists in registry
        if transformation_group is not None and transformation_group != 'self':
            if transformation_group not in self.registered_columns:
                raise MissingTransformationReferral(
                    column_name, 
                    transformation_group
                    )

        # Create the column entry
        entry = ColEntry(column_name            = column_name,
                         column_type            = column_type,
                         transformation         = transformation,
                         transformation_group   = transformation_group,
                         transformation_params  = transformation_params)
        
        # Append to the registry
        self._entries.append(entry)
        logger.debug("ColEntry '%s' added.", column_name)                  
    
    def update_transformation(self, 
                              column_name : str, 
                              params : LogParams | ZScoreParams | MinMaxParams) -> None:
        """
        Adjust the transformation_params of a ColEntry that may or may not already 
        exist.

        Parameters
        ----------
        column_name : str
            Name under which column is saved in ``ColumnRegistry``.
        params : LogParams | ZScoreParams | MinMaxParams
            Parameters to be saved at this ``ColEntry``.
        """
        col = self.get_entry_by_name(column_name)

        if col.transformation_params is None:
            col.transformation_params = TransformationParams()

        # match the type of the parameters, and if these have already been set, then throw an error
        match params:

            case LogParams():
            
                if col.transformation_params.log is not None:
                    raise TransformationParamsAlreadySet(
                        column_name, 
                        params.__class__.__name__
                        )
            
                col.transformation_params.log = params
                
            case ZScoreParams():

                if col.transformation_params.zscore is not None:
                    raise TransformationParamsAlreadySet(
                        column_name, 
                        params.__class__.__name__
                        )
            
                col.transformation_params.zscore = params

            case MinMaxParams():
                if col.transformation_params.minmax is not None:
                    raise TransformationParamsAlreadySet(
                        column_name, 
                        params.__class__.__name__
                        )
            
                col.transformation_params.minmax = params                            

            case _:
                raise ValueError(f"Unsupported params type: {type(params)}")
            
        logger.debug("ColEntry '%s' _transformation_params transform_params updated.", 
                     column_name)                  
        
    # ========= INTERACTING ======= #        
    def get_entries_names_by_type(self, column_type: str) -> list[str]:
        """Get all column names of a specific type"""
        return [col.column_name for col in self._entries if col.column_type == column_type]
    
    def get_entries_by_type(self, column_type: str) -> list[ColEntry]:
        """Get all ``ColEntry`` instances of a specific type"""
        return [col for col in self._entries if col.column_type == column_type]
    
    def get_transformation_groups(self) -> dict[str, list[str]]:
        """
        Get columns grouped by their normalization reference.
        
        Returns:
        --------
        dict : {normalization_group: [column_names]}
            Keys are the reference columns (is the column name itself, when 
            ``transformation_group`` == ``self``.)
            Values are lists of columns that share that normalization
        """
        groups: dict[str, list[str]] = {}
        for entry in self._entries:
            if entry.transformation:

                # Use the column itself as key if normalization_group is 'self'
                if entry.transformation_group:
                    key = entry.transformation_group if entry.transformation_group != 'self' else entry.column_name
                else:
                    key = 'undefined'

                if key not in groups:
                    groups[key] = []

                groups[key].append(entry.column_name)

        return groups
    
    def get_entry_by_name(self, column_name: str) -> ColEntry:
        """Get a specific column entry by name"""
        for col in self._entries:
            if col.column_name == column_name:
                return col
            
        raise MissingColEntry(column_name)

    # ========= PROPERTIES ======= #
    @property
    def context_columns(self) -> list[str]:
        """Get context column names"""
        return self.get_entries_names_by_type('context')
    
    @property
    def feature_columns(self) -> list[str]:
        """Get feature column names"""
        return self.get_entries_names_by_type('feature')
    
    @property
    def target_columns(self) -> list[str]:
        """Get target column names"""
        return self.get_entries_names_by_type('target')
    
    @property 
    def pred_columns(self) -> list[str]:
        """Get pred column names"""
        return self.get_entries_names_by_type('pred')        

    @property
    def split_columns(self) -> list[str]:
        """Get split column names"""
        return self.get_entries_names_by_type('split')
    
    @property 
    def registered_columns(self) -> list[str]:
        return [col.column_name for col in self._entries]        

    # ========= REPRESENTATION ======= #    
    def __repr__(self) -> str:
        type_counts = {
            'context'   : len(self.context_columns),
            'feature'   : len(self.feature_columns),            
            'target'    : len(self.target_columns),
            'pred'      : len(self.pred_columns),
            'split'     : len(self.split_columns)
        }

        representation = (            
            f"<{self.__class__.__name__}("+
            f"registered_columns = {sum(type_counts.values())}, "+
            ", ".join([f"{key} = {value}" for key,value in type_counts.items()]) +
            ")>"      
            )
        
        return representation
    
    def __str__(self) -> str:
        all_keys        = ['context','feature','split','target','pred']
        width           = max(len(k) for k in all_keys)
        indent          = 4
        
        lines = [f"<{self.__class__.__name__}("]     
        lines.append(align('context', self.context_columns, width, indent))
        lines.append(align('feature', self.feature_columns, width, indent))
        lines.append(align('target',  self.target_columns,  width, indent))
        lines.append(align('pred',    self.pred_columns,    width, indent))
        lines.append(align('split',   self.split_columns,   width, indent))        
        
        lines.append(")>")
        representation = '\n'.join(lines)
        return representation