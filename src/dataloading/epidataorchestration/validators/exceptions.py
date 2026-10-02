
class UnexpectedAttributeTypeError(Exception):
    
    def __init__(self, attribute_name: str, cls_name: str, found_type: str, allowed_types: list[str]):
        message = f"Attribute {attribute_name} has an unexpected type in {cls_name}. Got {found_type} but only {allowed_types} are allowed"
        super().__init__(message)

class EmptyAttributeTypeError(Exception):
    
    def __init__(self, attribute_name: str, cls_name: str):
        message = f"Attribute {attribute_name} in {cls_name} is empty"
        super().__init__(message)

class MissingColumnError(Exception):
    
    def __init__(self, attribute_name: str, column_name: str, cls_name: str):
        message = f"Attribute {attribute_name} in {cls_name} has no column {column_name}"
        super().__init__(message)        

class InvalidTokenizationError(Exception):
    
    def __init__(self, attribute_name: str, cls_name: str, missing_tokens: list[int], leftover_tokens: list[int]):
        message = f"Attribute {attribute_name} in {cls_name} is not properly tokenized. Missing tokens: {missing_tokens}. Leftover tokens: {leftover_tokens}"
        super().__init__(message)        

class NaNsFoundError(Exception):
    
    def __init__(self, attribute_name: str, cls_name: str, columns: list[str]):
        message = f"Attribute {attribute_name} in {cls_name} has NaNs in columns {columns}."
        super().__init__(message)        

class InvalidNormalizationError(Exception):

    def __init__(self, attribute_name: str, cls_name: str, column: str, specifications: str):
        message = f"Attribute {attribute_name} in {cls_name} has an invalid normaliztaion for {column}.\n{specifications}"
        super().__init__(message)        

class IncorrectPeakTimeError(Exception):

    def __init__(self, cls_name: str, specifications: str):
        message = f"Target peak time in {cls_name} is incorrect.\n{specifications}"
        super().__init__(message)  