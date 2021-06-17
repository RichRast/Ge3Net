from src.models import Model_A, Model_B, Model_C

class modelSelect():
    _models={
        "Model_A": lambda params, criterion, cp_criterion  : Model_A.model_A(params, criterion, cp_criterion),
        "Model_B": lambda params, criterion, cp_criterion : Model_B.model_B(params, criterion, cp_criterion),
        "Model_C": lambda params, criterion, cp_criterion : Model_C.model_C(params, criterion, cp_criterion)
    }

    @classmethod
    def get_selection(cls):
        return {
            'models':cls._models
        }