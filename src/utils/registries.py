from typing import Callable, List
import inspect

def registry_method(func: Callable) -> Callable:
    """
    Decorator that marks a method as a registered one.
    Used to create a selection of methods for graph-building
    and graph-normalization.
    """
    setattr(func, '_is_registered_method', True)
    return func

def get_registered_methods(cls: type) -> List[str]:
    """
    Returns the names of all methods decorated with @registry_method
    for a given class.
    """
    return [
        name for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
        if getattr(func, '_is_registered_method', False)
    ]