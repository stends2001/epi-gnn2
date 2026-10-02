class InvalidGraphStructure(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)

class InvalidGraphObject(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)