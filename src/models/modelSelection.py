from src.models import Model_A, Model_B, Model_C, Model_D, Model_E, Model_F, Model_G, \
Model_H, Model_I, Model_J, Model_K, Model_L, Model_M, Model_N

class modelSelect():
    _models={
        "Model_A": lambda params, criterion, cp_criterion  : Model_A.model_A(params, criterion, cp_criterion),
        "Model_B": lambda params, criterion, cp_criterion : Model_B.model_B(params, criterion, cp_criterion),
        "Model_C": lambda params, criterion, cp_criterion : Model_C.model_C(params, criterion, cp_criterion),
        "Model_D": lambda params, criterion, cp_criterion : Model_D.model_D(params, criterion, cp_criterion),
        "Model_E": lambda params, criterion, cp_criterion : Model_E.model_E(params, criterion, cp_criterion),
        "Model_F": lambda params, criterion, cp_criterion : Model_F.model_F(params, criterion, cp_criterion),
        "Model_G": lambda params, criterion, cp_criterion : Model_G.model_G(params, criterion, cp_criterion),
        "Model_H": lambda params, criterion, cp_criterion : Model_H.model_H(params, criterion, cp_criterion),
        "Model_I": lambda params, criterion, cp_criterion : Model_I.model_I(params, criterion, cp_criterion),
        "Model_J": lambda params, criterion, cp_criterion : Model_J.model_J(params, criterion, cp_criterion),
        "Model_K": lambda params, criterion, cp_criterion : Model_K.model_K(params, criterion, cp_criterion),
        "Model_L": lambda params, criterion, cp_criterion : Model_L.model_L(params, criterion, cp_criterion),
        "Model_M": lambda params, criterion, cp_criterion : Model_M.model_M(params, criterion, cp_criterion),
        "Model_N": lambda params, criterion, cp_criterion : Model_N.model_N(params, criterion, cp_criterion),
    }

    @classmethod
    def get_selection(cls):
        return {
            'models':cls._models
        }