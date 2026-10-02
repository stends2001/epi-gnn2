from collections import defaultdict
from collections.abc import Sequence 

class ExceptionReport(Exception):
    """
    Aggregates multiple exception - classes into a single, user-friendly report.
    """
    def __init__(self, 
                 exceptions:    Sequence[Exception], 
                 context:       str):
        
        self.exceptions = exceptions
        self.context    = context
        self._exceptions_dict = self._organize_exception_types()

        super().__init__(self.__str__())

    def _organize_exception_types(self) -> dict:
        types = defaultdict(list)
        for e in self.exceptions:
            types[type(e)].append(e)
        return dict(types)

    def __str__(self) -> str:
        separation_line = "-"*50
        numbers_per_type = []
        error_messages = []

        for exc_type, exc_list in self._exceptions_dict.items():
            error_messages.append(separation_line)
            count = len(exc_list)
            numbers_per_type.append(f"{count} {exc_type.__name__}{'' if count == 1 else 's'}")
            
            error_messages.append(f"{exc_type.__name__}:")
            for idx, exc in enumerate(exc_list):
                error_messages.append(f"   [{idx+1}] {exc}")

        error_messages.append(separation_line)
        startline = f"{self.context} by {', '.join(numbers_per_type)}:\n"
        return startline + "\n".join(error_messages)