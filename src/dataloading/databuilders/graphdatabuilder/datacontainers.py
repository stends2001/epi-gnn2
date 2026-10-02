import torch

from ....graphconstruction import GraphStructure

class DataEntryError(Exception):
    """
    Wrong data entry inside GraphDataList or DeepDataList
    """    
    def __init__(self, message: str):
        super().__init__(message)

class Data:
    """
    An alternative to the Pytorch Data class for data-entries of X, y

    Parameters
    ----------
    x: Tensor
        input data of shape [num_nodes, num_features, seq_len]
    y: Tensor
        target data of shape [num_nodes, horizon_size]
    """
    def __init__(self, 
                 x:     torch.Tensor, 
                 y:     torch.Tensor,
                 graph: GraphStructure | None = None):
        
        self.x          = x
        self.y          = y
        self.graph      = graph

        self.__post_init_validate()

    def __post_init_validate(self) -> None:
        """validate entry against expected shapes"""
        
        if self.x.dim() != 3:
            raise DataEntryError(f"x must be 3D [num_nodes, seq_len, num_features], got {self.x.dim()}D")
        
        if self.y.dim() != 2:
            raise DataEntryError(f"y must be 2D [num_nodes, horizon_size], got {self.y.dim()}D")
        
        if self.graph is not None and self.graph.num_nodes != self.x.shape[0]:
                raise DataEntryError(f"num_nodes in graph is {self.graph.num_nodes} but got features for {self.x.shape[0]} nodes.")


    def to(self, device: torch.device) -> 'Data':
        """Move all tensors to the specified device (GPU)"""
        return Data(
            x=self.x.to(device),
            y=self.y.to(device),
            graph = None if self.graph is None else self.graph.to(device)
            )        
    
    def __repr__(self):
        x_repr      = f"x: {tuple(self.x.shape)}"
        y_repr      = f"y: {tuple(self.y.shape)}"
        graph_repr  = "" if self.graph is None else str(self.graph)

        representation = f"<{self.__class__.__name__}({x_repr}, {y_repr}, {graph_repr})>"

        return representation
      
class DataList:
    """
    An alternative to the Pytorch DataLoader class
    Basically a list of GraphData entries

    Extension to DeepData, but with a graphstructure    
    """    
    def __init__(self, data_list: list[Data]):
        self.data_list = data_list

        self._post_init_validate()

    def _post_init_validate(self) -> None:
        """validate entry against expected shapes"""
       
        if len(self.data_list) == 0:
            raise DataEntryError(f"got empty data_list")

        elif not isinstance(self.data_list[0], Data):
            raise DataEntryError(f"got unexpected type {self.data_list[0].__class__.__name__}")

    def __iter__(self):
        return iter(self.data_list)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]
    
    def __repr__(self):
        representation = f"<{self.__class__.__name__}({len(self.data_list)} datapoints , sample: {str(self.data_list[0])})>"        
        return representation  
