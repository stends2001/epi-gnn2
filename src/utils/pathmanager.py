from pathlib import Path

from .exceptions import PathNotFound

class PathManager:
    """ 
    Manages paths in this projects

    Attributes
    ----------
    - `project_root`
    - `data`
    - `src`
    - `outcomes`
    - `results`
    """
    def __init__(self):
        self.project_root   = Path(__file__).resolve().parent.parent.parent 
        self.data           = self.project_root / 'data'
        self.src            = self.project_root / 'src'
        self.outcomes       = self.project_root / 'outcomes'
        self.results        = self.project_root / 'results'
        self.figures        = self.project_root / 'results' / 'figures'

        self._validate_paths()
        self._setup_paths()

    def _setup_paths(self):
        """Create output directories"""
        for path in (self.outcomes, self.results, self.figures):
            path.mkdir(parents=True, exist_ok=True)          

    def _validate_paths(self):
        """Validate required paths"""

        readme = self.project_root / "README.md"

        if not readme.is_file():
            raise PathNotFound(readme)

        for path in (self.data, self.src):
            if not path.is_dir():
                raise PathNotFound(path)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"