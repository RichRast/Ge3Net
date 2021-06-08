import torch
from torch import nn
import torch.nn.functional as F
from src.main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss,\
    class_accuracy, computePrMetric
from src.utils.modelUtil import Swish
from src.models import Model_A, Model_B

class Selections():
    _losses={
        'gcd': lambda reduction, alpha, geography: GcdLoss() if geography else None,
        'l1_loss': lambda reduction, alpha, geography: torch.nn.L1Loss(reduction=reduction),
        'mse': lambda reduction, alpha, geography: torch.nn.MSELoss(reduction=reduction),
        'smooth_l1': lambda reduction, alpha, geography: SmoothL1Loss(reduction=reduction, alpha=alpha),
        'weighted_loss': lambda reduction, alpha, geography: Weighted_Loss(reduction=reduction, alpha=alpha)
    }
    _balancedMetrics={
        'median': lambda metricsObj: metricsObj.medianMetric,
        'meanBalanced': lambda metricsObj: metricsObj.meanBalanced, 
        'medianBalanced': lambda metricsObj:metricsObj.medianBalanced
    }
    _cpMetrics={
        'loss_cp':F.binary_cross_entropy_with_logits, 
        'prMetric':computePrMetric
    }
    _spMetrics={
        'loss_sp': F.cross_entropy, 
        'Accuracy': class_accuracy, 
        'Precision': None, 
        'Recall': None, 
        'BalancedAccuracy': None
    }
    _normalizationLayers={
        'LayerNorm': lambda input_units: nn.LayerNorm(input_units),
        'BatchNorm': lambda input_units: nn.BatchNorm1d(input_units),
        'GroupNorm': lambda input_units: nn.GroupNorm(input_units),
    }
    _activation={
        'swish':lambda p: nn.Swish(),
        'relu': lambda p: nn.ReLU(),
        'leakyRelu':lambda p: nn.LeakyReLU(p),
        'gelu':lambda p: nn.GELU(),
        'selu': nn.SELU(),
        'sigmoid':nn.Sigmoid(),
        'softplus': nn.Softplus()
    }
    _weightInitializations={
        'kiaming_normal': lambda weight: nn.init.kaiming_normal_(weight),
        'lecun_normal': None 
    }
    _dropouts={
        'dropout':lambda p : nn.Dropout(p),
        'alphaDropout': lambda p: nn.AlphaDropout(p)
    }
    _models={
        "Model_A": lambda params : Model_A.modelA(params),
        "Model_B": lambda params : Model_B.modelB(params)
    }

    @classmethod
    def get_selection(cls):
        return {
            'loss':cls._losses,
            'balancedMetrics':cls._balancedMetrics,
            'cpMetrics':cls._cpMetrics,
            'spMetrics':cls._spMetrics,
            'normalizationLayer':cls._normalizationLayers,
            'activation':cls._activation,
            'dropouts':cls._dropouts,
            'models':cls._models
        }