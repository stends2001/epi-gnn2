from typing import Union 
from pathlib import Path

class PathNotFound(Exception):
    def __init__(self, path: Union[str, Path]):
        super().__init__(f'Path was not found: {path}')

class InvalidExtension(Exception):
    def __init__(self, extension_expected: str, extension_got: str):
        msg = f"Expected' {extension_expected}', got '{extension_got}'."
        super().__init__(msg)
