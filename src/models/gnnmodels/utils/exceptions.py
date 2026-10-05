class InvalidOptimizerError(Exception):
    def __init__(self, optimizer_name: str, supported_optimizers: list[str]):
        message = f'Invalid optimizer {optimizer_name}. Supported optimzers are {supported_optimizers}'
        super().__init__(message)        

class InvalidLossError(Exception):
    def __init__(self, loss_name: str, supported_losses: list[str]):
        message = f'Invalid loss {loss_name}. Supported losses are {supported_losses}'
        super().__init__(message)         

class InvalidSchedulerError(Exception):
    def __init__(self, scheduler_name: str, supported_schedulers: list[str]):
        message = f'Invalid scheduler {scheduler_name}. Supported schedulers are {supported_schedulers}'
        super().__init__(message)      

class UnexpectedDataShape(Exception):
    """
    errors are to be raised!
    """    
    def __init__(self, received_obj: str, expected_obj: str, context: str):
        message = f"context: {context} \nExcpected {expected_obj}, got {received_obj}"
        super().__init__(message)

class InconsistentDataShape(Exception):
    def __init__(self, message: str):
        super().__init__(message)    