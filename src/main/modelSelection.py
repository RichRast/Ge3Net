import torch
import torch.nn.functional as F
from src.main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss, eval_cp_batch,\
    class_accuracy, computePrMetric, t_prMetrics

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
        'LayerNorm': lambda input_units: torch.nn.LayerNorm(input_units),
        'BatchNorm': lambda input_units: torch.nn.BatchNorm1d(input_units),
        'GroupNorm': lambda input_units: torch.nn.GroupNorm(input_units),
    }

    @classmethod
    def get_selection(cls):
        return {
            'loss':cls._losses,
            'balancedMetrics':cls._balancedMetrics,
            'cpMetrics':cls._cpMetrics,
            'spMetrics':cls._spMetrics,
            'normalizationLayer':cls._normalizationLayers
        }