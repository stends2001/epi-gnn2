
class GraphRegistryMessage:
    """Parent update class"""
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"

class GraphEntryAlreadyExists(GraphRegistryMessage):

    def __init__(self, graphname: str):
        message = f"Graphname {graphname} is already in GraphRegistry. The new graph is not saved."
        super().__init__(message)

class GraphEntryRegistered(GraphRegistryMessage):

    def __init__(self, graphname: str):
        message = f"Graphname {graphname} is registered"
        super().__init__(message)

class GraphEntryRemoved(GraphRegistryMessage):

    def __init__(self, graphname: str):
        message = f"Graphname {graphname} removed from GraphRegistry."
        super().__init__(message)

class InvalidGraphName(GraphRegistryMessage):

    def __init__(self, graphname: str, allowed_characters: set):
        message = f"Graphname {graphname} is invalid. Please stick with the following_characters: {allowed_characters}."
        super().__init__(message)

class GraphStructureSaved(GraphRegistryMessage):

    def __init__(self, graphname: str):
        message = f"Graphname {graphname} is saved."
        super().__init__(message)

class GraphEntryDoesntExist(GraphRegistryMessage):

    def __init__(self, graphname: str, registered_entries: list[str]):
        message = f"Graphname {graphname} does not exist in GraphRegistry. Registered entries are {registered_entries}"
        super().__init__(message)

class GraphStructureAlreadySaved(GraphRegistryMessage):

    def __init__(self, graphname: str):
        message = f"Graphstructure with graphname {graphname} is already saved."
        super().__init__(message)
# =========   EXCEPTIONS   =========== #
class InvalidTokenizationMapFound(Exception):

    def __init__(self, graphname: str):
        message = f"Tokenization map found at {graphname} is different to that found in the graph directory!"
        super().__init__(message)

class NoTokenizationMapFound(Exception):

    def __init__(self, graph_dir: str):
        message = f"No tokenization map found in {graph_dir}"
        super().__init__(message)

class NoGraphStructureFound(Exception):

    def __init__(self, graphname: str, graph_dir: str):
        message = f"No graph {graphname} found in {graph_dir}"
        super().__init__(message)