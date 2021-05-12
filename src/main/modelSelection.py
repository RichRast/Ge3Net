import torch
from src.main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss

class Selections():
    _losses={
        'gcd': lambda reduction, alpha: GcdLoss(),
        'l1_loss': lambda reduction, alpha: torch.nn.L1Loss(reduction=reduction),
        'mse': lambda reduction, alpha: torch.nn.MSELoss(reduction=reduction),
        'smooth_l1': lambda reduction, alpha: SmoothL1Loss(reduction=reduction, alpha=alpha),
        'bce': lambda reduction, alpha: torch.nn.BCEWithLogitsLoss(reduction=reduction),
        'weighted_loss': lambda reduction, alpha: Weighted_Loss(reduction=reduction, alpha=alpha),
    }

    @classmethod
    def get_selection(cls):
        return {
            'loss':cls._losses
        }