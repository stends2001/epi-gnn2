# ======= EPICONFIG ======= #
class EpiConfigValidationError(Exception):   
    def __init__(self, msg: str):
        super().__init__(msg)

class EpiConfigLimitationError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)      

class IncompatibleEpiConfigs(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)