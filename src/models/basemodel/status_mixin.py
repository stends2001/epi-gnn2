from ..utils import ModelStatusError, ModelStatus

class ModelStatusMixin:
    """ 
    Mixin class to ``BaseModel`` that deals with the model's status.
    """
    status_dict: dict[ModelStatus, bool]

    def _init_status(self):
        """sets attribute status_dict with all values False"""

        self.status_dict = {
            'model_initialized' : False,
            'model_hparams_set' : False,
            'global_hparams_set': False,
            'trained'           : False,
            'forecasted'        : False,
        }

    def _update_status(self, status: ModelStatus):
        """
        updates a key-value in self.status_dict
        """
        self.status_dict[status] = True 

    def _check_status(self, required_states: list[ModelStatus] | ModelStatus):
        """
        Validates that required setup steps have been completed
        """
        if isinstance(required_states, str):
            required_states = [required_states]

        missing = [state for state in required_states if not self.status_dict.get(state, False)]

        if missing:
            raise ModelStatusError(
                f"Missing required setup steps: {', '.join(missing)}. "
                f"Call the corresponding methods first."
            )