from typing import Literal 

CRS_DEGREES = "EPSG:4326"
CRS_METRES  = "EPSG:25833"

GraphType       = Literal['identity','geographical_contiguity','gravity_model','random','fully_connected']
GraphNormType   = Literal['minmax','symmetric','rowwise','none']