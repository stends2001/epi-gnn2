from typing import Optional, Self, Tuple, List, Literal, cast
import torch
import numpy as np
import pandas as pd
import os

from .datacontainers import DataList, Data
from ...epidataorchestration.orchestrator import EpiDataOrchestrator
from ....utils import checkmark
from ....graphconstruction import GraphObject, GraphStructure

class DataListError(Exception):
    def __init__(self, message: str):
        super().__init__(message)   

class GraphMissingError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class GraphDataBuilder:
    """
    Base class for creating deep-dataloaders from EpiDataOrchestrator.
    Handles X/y construction and temporal splitting.
    """
    def __init__(self, 
                 dataorchestrator: EpiDataOrchestrator):
        
        self.dataorchestrator       = dataorchestrator
        self.column_registration    = dataorchestrator.column_registration

        self.graph: Optional[GraphStructure]                  = None
        self._graphmode: Optional[Literal['static','dynamic']] = None

        self._dataloader_main: Optional[DataList]    = None
        self._dataloader_train: Optional[DataList]   = None
        self._dataloader_val: Optional[DataList]     = None
        self._dataloader_test: Optional[DataList]    = None     
    
    def build(self) -> Self:
        """
        Orchestrates the entire GraphDataLoaderManager - creation. 

        First retrieve the graph structure
        """

        X, y     = self._split_Xyt(self.dataorchestrator.data_final.data)

        main, train, val, test = cast(
            tuple[list[Data], list[Data], list[Data], list[Data]],
            self._build_sequences(X, y)
        )
                
        # Wrap in simple lists or custom container
        self._dataloader_main    = DataList(main)
        self._dataloader_train   = DataList(train)
        self._dataloader_val     = DataList(val)
        self._dataloader_test    = DataList(test)        
        return self

    def _split_Xyt(self, df: pd.DataFrame) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        ConstructsTensor objects from long-format dataframe.
        Equal for any sublcass
        
        Parameters
        ----------
        df : pd.DataFrame
            Long-format data with columns: [{temporal_column}, {id_column}, {features}, {targets}, splits]
        
        Returns
        -------
        X : Tensor
            Input data of shape [num_timestamps, num_nodes, num_features]
        y : Tensor
            Target data of shape [num_timestamps, num_nodes, num_targets]

        Also creates

        time_splits : pd.DataFrame
            DataFrame with columns [timestamp, train, val, test] and t_idx as index
        """
        
        # Get column groups
        feature_cols    = self.column_registration.get_entries_names_by_type('feature')
        split_cols      = self.column_registration.get_entries_names_by_type('split')
        target_cols     = self.column_registration.get_entries_names_by_type('target')
        
        temporal_col    = self.dataorchestrator.config.temporal_column
        id_col          = self.dataorchestrator.config.id_column
        
        # Store time splits for later use with t_idx as index
        time_splits = (
            df[[temporal_col] + split_cols]
            .drop_duplicates()
            .sort_values(temporal_col)
            .reset_index(drop=True)
        )
        time_splits.index.name  = 't_idx'
        self.time_splits        = time_splits
        
        # Helper function to pivot and convert columns
        def pivot_columns(columns: List[str]) -> np.ndarray:
            """Pivot columns from long to wide format and stack into 3D array"""
            arrays = []
            
            for col in columns:
                # Pivot: rows=time, cols=nodes, values=column
                wide = df.pivot(
                    index=temporal_col,
                    columns=id_col,
                    values=col
                )
                
                # Convert to numeric, then to numpy array
                arr = wide.apply(pd.to_numeric, errors='coerce').values
                
                # Preserve int types, otherwise use float32
                if df[col].dtype.kind == 'i':
                    arr = arr.astype(np.int32)
                else:
                    arr = arr.astype(np.float32)
                
                arrays.append(arr)
            
            # Stack: [time, nodes, features]
            return np.stack(arrays, axis=-1)
        
        # Build X from feature columns
        X_np = pivot_columns(feature_cols)
        
        # Build y from target columns (any column containing target names but not in features)
        target_column_names = [
            col for col in df.columns
            if any(target in col for target in target_cols)
            and col not in feature_cols
        ]
        
        y_np = pivot_columns(target_column_names)
        
        # Convert to tensors
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        
        return X, y
    
    def _build_sequences(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[List[Data], 
                                                              List[Data], 
                                                              List[Data], 
                                                              List[Data]]:
        """
        Creates sequences from X, y tensors.
        Returns lists of data objects (not wrapped in container yet).
        Equal for any subclass
        """
        # main contains train+val+test
        dataset_main:  List[Data] = []
        dataset_train: List[Data] = []
        dataset_val:   List[Data] = []
        dataset_test:  List[Data] = []

        # number of timesteps
        T       = X.shape[0]
        min_end = self.dataorchestrator.config.sequence_length
        max_end = T + 1

        if min_end > T:
            raise ValueError(f'{self.__class__.__name__} couldnt build sequences as min_end > T; {min_end} > {T}')

        for x_end in range(min_end, max_end):
            x_start = x_end - self.dataorchestrator.config.sequence_length
            x_seq = X[x_start : x_end]
            y_seq = y[x_end - 1]
            target_idx = x_end - 1
            
            # Create data object (to be overridden by subclasses)
            data = self._create_data_object(x_seq, y_seq)
            
            # Split assignment
            if self.time_splits.loc[target_idx, 'train']:
                dataset_train.append(data)
            elif self.time_splits.loc[target_idx, 'val']:
                dataset_val.append(data)
            elif self.time_splits.loc[target_idx, 'test']:
                dataset_test.append(data)
            
            dataset_main.append(data)

        return dataset_main, dataset_train, dataset_val, dataset_test
    
    def _create_data_object(self, x_seq: torch.Tensor, y_seq: torch.Tensor) -> 'Data':
        """Create GraphData object with graphstructure"""
        return Data(
            x = x_seq.clone().detach().float().permute(1, 2, 0),
            y = y_seq.clone().detach().float(),
            graph = self.graph
        )
    
    def retrieve_static_graph(self, graphname: str) -> 'GraphDataBuilder':
        """retrieves a static (opposed to dynamic) graph structure"""

        graphdir_parent = self.dataorchestrator.config.path_manager.tokenization_map.parent

        graphpath   = os.path.join(str(graphdir_parent), graphname)
        graphobject = GraphObject.load(graphpath)
        
        graph_structure = graphobject.graph

        self.graph      = graph_structure
        self._graphmode = 'static'
        return self

    @property 
    def dataloader_main(self) -> 'DataList':
        if self._dataloader_main is None:
            raise DataListError('dataloader_main not found')
        else:
            return self._dataloader_main
        
    @property 
    def dataloader_train(self) -> 'DataList':
        if self._dataloader_train is None:
            raise DataListError('dataloader_train not found')
        else:
            return self._dataloader_train

    @property 
    def dataloader_val(self) -> 'DataList':
        if self._dataloader_val is None:
            raise DataListError('dataloader_val not found')
        else:
            return self._dataloader_val

    @property 
    def dataloader_test(self) -> 'DataList':
        if self._dataloader_test is None:
            raise DataListError('dataloader_test not found')
        else:
            return self._dataloader_test                        

    def __repr__(self) -> str:
        parts = []

        if self._dataloader_main:
            parts.append(f"dataloader_main {checkmark}")
        if self._dataloader_train:
            parts.append(f"dataloader_train {checkmark}")
        if self._dataloader_val:
            parts.append(f"dataloader_val {checkmark}")
        if self._dataloader_test:
            parts.append(f"dataloader_test {checkmark}")

        inner = ', '.join(parts) if parts else 'not built'
        return f"<{self.__class__.__name__}({inner})>"