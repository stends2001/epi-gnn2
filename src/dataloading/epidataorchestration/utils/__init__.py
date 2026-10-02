berlin_district_ids = ['11001', '11002', '11003', '11004', '11005', '11006', '11007', '11008', '11009', '11010', '11011', '11012']
berlin_id           = '11000'

from .exceptions import EpiDataOrchestrationError, MissingEpiDataContainer, NonExistentAttributeEpiDataContainer, TemporalError
from .temporal_summary import EpiDataTemporalSummary