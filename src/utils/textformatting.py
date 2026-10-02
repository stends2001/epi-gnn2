from typing import Dict, Any, Union, List, Tuple

checkmark       = '✓'
crossmark       = '✗'

def align(key: str, value: Any, width: int, indent: int = 4, newline: bool = False) -> str:
    """
    Format a single key-value line with alignment.
    
    Parameters
    ----------
    key : str
        The key/label
    value : Any
        The value to display
    width : int
        Width for key alignment
    indent : int
        Number of spaces for indentation
    newline : bool
        whether or not to return the line with "\n" 
        
    Returns
    -------
    str : Formatted line
    
    Examples
    --------
    >>> align('name', 'Claude', 15)
    '    name            : Claude'
    >>> align('trained', True, 15)
    '    trained         : True'
    """
    spaces = ' ' * indent

    line   = f"{spaces}{key:<{width}} : {value}"

    if newline:
        line+="\n"

    return line

def return_header_line(header: str, indent_char: str = "=", n_indent_chars: int = 8, n_space: int = 3, indent: int = 0):
    line = " "*indent + indent_char*n_indent_chars + " " *n_space + header.upper() + " "*n_space + indent_char*n_indent_chars
    return line

def section(title: str, 
            items: Union[Dict[str, Any], List[Tuple[str, Any]]], 
            width: int, 
            indent: int = 4,
            separator: str = '-') -> List[str]:
    """
    Format a section with header and items.
    
    Parameters
    ----------
    title : str
        Section title (will be uppercased)
    items : Union[Dict[str, Any], List[Tuple[str, Any]]]
        Items to display as key-value pairs
    width : int
        Width for alignment
    indent : int
        Number of spaces for indentation
    separator : str
        Character for section header separator
        
    Returns
    -------
    List[str] : List of formatted lines (header + items)
    
    Examples
    --------
    >>> section('status', {'trained': True, 'tested': False}, 15)
    ['    -----STATUS------', '    trained         : True', '    tested          : False']
    >>> section('info', [('name', 'Claude'), ('age', 2)], 10)
    ['    ---INFO----', '    name       : Claude', '    age        : 2']
    """
    spaces = ' ' * indent
    header = f"{spaces}{title.upper().center(width + 3, separator)}"
    
    if isinstance(items, dict):
        items = list(items.items())
    
    lines = [header]
    for key, value in items:
        lines.append(align(key, value, width, indent))
    
    return lines