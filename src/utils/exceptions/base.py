from typing import List

class AttributeNotFound(Exception):
    def __init__(self, missing_attr: str, class_name: str):
        msg = f"no attribute {missing_attr} in {class_name}"
        super().__init__(msg)

class MissingColumnError(Exception):
    def __init__(self, col: str, df_name: str):
        msg = f'column {col} not found in {df_name}'
        super().__init__(msg)

class InvalidDataSetError(Exception):
    def __init__(self, input: str):
        super().__init__(f"Invalid value for dataset: {input}. Expected one of ['train','val','test']")    

class UnequalSetsError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class MethodNotInRegistry(Exception):
    def __init__(self, method: str, available_methods: List[str]):
        msg = f'Unknown method {method}. Available methods are: {available_methods}'
        super().__init__(msg)
