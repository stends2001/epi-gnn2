class EpiDataOrchestrationError(Exception):
    def __init__(self, msg: str):
        msg = f"Data Orchestration couldn't be run; {msg}"
        super().__init__(msg)    

class MissingEpiDataContainer(Exception):
    def __init__(self, datastage: str, previous_method: str):
        msg = f"No {datastage} attribute found for DataOrchestrator. Run {previous_method}() first"
        super().__init__(msg)     

class NonExistentAttributeEpiDataContainer(Exception):
    def __init__(self, class_name: str, attribute_name: str):
        msg= f"Attribute {attribute_name} does not exist in {class_name}"
        super().__init__(msg)               

# ======= TEMPORAL SUMMARY ==== #
class TemporalError(Exception):
    def __init__(self, message: str):
        msg = f'Invalid EpiDataTemporalSummary: {message}'
        super().__init__(msg)