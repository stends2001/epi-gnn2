class MissingColEntry(Exception):
    """
    Raised when looking for a column entry that doesn't exist
    """
    def __init__(self, entryname: str):
        message = f"Entry '{entryname}' doesn't exist in ColRegistry."
        super().__init__(message)        

class MissingTransformationReferral(Exception):
    """
    Raised when a transformation referral is invalid
    """
    def __init__(self, entryname: str, referral: str):
        message = f"Entry '{entryname}' has transformation_group {referral} for which no transformation attribute was found."
        super().__init__(message)

class TransformationParamsAlreadySet(Exception):
    """
    Raised when a transformation params is adjusted while already existent
    """
    def __init__(self, entryname: str, transformation_type: str):
        message = f"Entry '{entryname}' already has a transformatoin_params of type {transformation_type}! Cannot be overwritten."
        super().__init__(message)

class ColEntryMissingAttribute(Exception):
    def __init__(self, entry_name: str, attribute_name: str):
        msg = f"Accession attempt was made on ColEntry {entry_name} for attribute {attribute_name} which is unavailable."
        super().__init__(msg)

class InvalidColEntry(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)    
