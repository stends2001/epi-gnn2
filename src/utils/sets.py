from typing import List, Any, Dict, Set
from .exceptions import UnequalSetsError

def compare_sets(set1: Set[Any], set2: Set[Any]):

    if set1 != set2:
        missing = set(set1) - set(set2)
        leftover= set(set2) - set(set1)

        raise UnequalSetsError(f'missing from set 2 are: {missing}. Leftover in set 1 are: {leftover}')
        
def reorder_dict(d: Dict[Any, Any], elements: List[str]) -> Dict[Any, Any]:
    """Reorder dictionary keys"""
    reordered = {}
    for key in elements:
        if key in d:
            reordered[key] = d[key]
    for key in d:
        if key not in reordered:
            reordered[key] = d[key]
    return reordered

