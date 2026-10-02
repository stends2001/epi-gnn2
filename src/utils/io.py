from typing import Optional, List, Dict, Union, Any
from pathlib import Path
import yaml
import pickle, json, copy

from .exceptions import PathNotFound

def load_mapping_dict(path: Union[str, Path]) -> Dict[Any, Any]:

    if isinstance(path, str):
        path = Path(path)

    match path.suffix:
        case ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)        
        case ".json":
            with open(path, "r") as f:
                data = json.load(f)
        case _:
            raise ValueError('the only type of files suppported in load_mapping_dict are ".pkl" and "json"')

    if type(data) != dict:
        raise ValueError(f'expected dictionary. got {type(data)}')
    return copy.deepcopy(data)

def save_mapping_dict(dictionary: Dict[Any,Any], path: Union[str, Path]):

    if isinstance(path, str):
        path = Path(path)

    if not path.parent.exists():
        raise PathNotFound(path.parent)

    match path.suffix:
        case ".pkl":
            raise ValueError('you should probably save this file as a .json instead!')
        case ".json":
            with open(path, "w") as f:
                json.dump(dictionary,f,indent = 4)
        case _:
            raise ValueError('the only type of files suppported in load_mapping_dict are ".pkl" and "json"')

def list_files(dir: Path, extension: Optional[str] = None) -> List[str]:
    """List files in directory with optional extension filter"""
    files_list = []
    for filepath in dir.iterdir():
        if filepath.is_file():
            if extension is None or filepath.suffix == extension:
                files_list.append(filepath.stem)
    return files_list

def write_yaml_file(yaml_dictionary: Dict, dir: Path, filename: str):
    """Write dictionary to YAML file"""
    if not filename.endswith('.yaml'):
        filename = filename + '.yaml'
    
    filepath = dir / filename
    with open(filepath, 'w') as file:
        yaml.dump(yaml_dictionary, file, sort_keys=False, default_flow_style=False)
