import numpy as np
import torch

from helper_funcs import get_gradient


def eval_cp_matrix(true_cps, pred_cps, seq_len = 317, win_tol=2):
    """
    For a given haplotype, compute the accuracy metrics for changepoint
    true_cp = np.array of true cp indices
    pred_cp = np.array of predicted cp indices
    win_tol = +/- tolerance for location and count of cp
    
    >>> eval_cp_matrix(true_cps=np.array([3,2]), pred_cps=np.array([4,1]))
    (1.0, 1.0, array([[1., 2.],[2., 1.]]))

    >>> eval_cp_matrix(true_cps=np.array([3,2]), pred_cps=np.array([4,1,5,6]))
    (0.6666666666666666, 1.0, array([[1., 2., 3., 4.],
       [2., 1., 2., 3.]]))
    """

    TP_count = 0
    FP_count = 0
    FN_count = 0
    n = len(true_cps)
    m = len(pred_cps)
    total_count = seq_len

    # assert statement to make sure fill_value > win_tol
    distance_matrix = np.full((n,m), fill_value=np.inf)

    if n==m==0:
        Precision, Recall, Accuracy, A_no_cp, Balanced_Accuracy = 1, 1, 1, 1, 1
        return Precision, Recall, Accuracy, A_no_cp, Balanced_Accuracy, distance_matrix
    elif n==0:
        Precision, Recall = 0, 1
        Accuracy = (total_count-m)/total_count
        A_no_cp = (total_count-m)/total_count
        Balanced_Accuracy = 0.5*(Recall + A_no_cp)
        return Precision, Recall, Accuracy, A_no_cp, Balanced_Accuracy, pred_cps
    elif m==0:
        Precision, Recall = 1, 0
        Accuracy = (total_count-n)/total_count
        A_no_cp = 1
        Balanced_Accuracy = 0.5*(Recall + A_no_cp)
        return Precision, Recall, Accuracy, A_no_cp, Balanced_Accuracy, true_cps

    #sort the true_cp and pred_cp list
    true_cp = sorted(true_cps)
    pred_cp = sorted(pred_cps)

    for j, val_b in enumerate(pred_cp):
            distance_matrix[:,j] = abs(true_cp - val_b)

    matches_mask = np.where(distance_matrix<=win_tol,0,1)
    
    FP_count = np.prod(matches_mask, axis=0).sum()
    FN_count = np.prod(matches_mask, axis=1).sum()
    TP_count = n-FN_count
    TN_count = total_count - (TP_count + FP_count + FN_count)
    Precision = TP_count/(TP_count + FP_count)
    Recall = TP_count/(TP_count + FN_count)
    Accuracy = (TP_count + TN_count)/total_count
    A_no_cp = TN_count/(TN_count + FP_count)
    Balanced_Accuracy = 0.5*(Recall + A_no_cp)

    return Precision, Recall, Accuracy, A_no_cp, Balanced_Accuracy, distance_matrix


def eval_cp_batch(cp_target, cp_pred, seq_len = 317, win_tol=2):
    """
    cp_pred : [Batch_size X T]
    1 for cp and 0 for no cp

    cp_target : [Batch_size X T]
    1 for cp and 0 for no cp

    """
    #Todo need to batchify this function and eval_cp_matrix func
    # and move tp tensors
    num_samples = cp_pred.shape[0]
    Precision, Recall = np.zeros((num_samples)), np.zeros((num_samples))
    Accuracy, A_no_cp, Balanced_Accuracy = \
        np.zeros((num_samples)), np.zeros((num_samples)), np.zeros((num_samples))
    distance_matrix=[]

    # convert to numpy
    if torch.is_tensor(cp_target):
        cp_target = cp_target.detach().cpu().numpy()
    if torch.is_tensor(cp_pred):
        cp_pred = cp_pred.detach().cpu().numpy()

    for i in range(num_samples):
        cp_target_idx = np.nonzero(cp_target[i,:])[0]
        cp_pred_idx = np.nonzero(cp_pred[i,:])[0]
        true_cps = cp_target_idx
        pred_cps = cp_pred_idx
        
        Precision[i], Recall[i], Accuracy[i], A_no_cp[i], Balanced_Accuracy[i], distance_matrix_tmp = \
            eval_cp_matrix(true_cps, pred_cps, seq_len, win_tol)
        distance_matrix.append(distance_matrix_tmp)
    
    return Precision.mean(), Recall.mean(), Accuracy.mean(), A_no_cp.mean(), Balanced_Accuracy.mean()


class SmoothL1Loss():
    def __init__(self, reduction='mean', beta=1.0):
        self.beta = beta
        self.reduction = reduction

    def __call__(self, input_y, target, device):
        tensor_1 = torch.tensor([1.0]).to(device)
        tensor_0 = torch.tensor([0.0]).to(device)
        mask = torch.where((input_y-target)<self.beta, tensor_1, tensor_0)
        rev_mask = torch.where((input_y-target)<self.beta, tensor_0, tensor_1)
        
        z1 = mask*0.5*(torch.pow((input_y-target),2)/self.beta)
        z2 = rev_mask*(abs(input_y - target) - 0.5*self.beta)
        z = z1 + z2
        if self.reduction=='sum':
            return z.sum().item()
        return z.mean().item()
    
class Weighted_Loss():
    def __init__(self, reduction='mean', alpha = 1.0):
        self.alpha = alpha
        self.reduction = reduction
        self.L1_loss = torch.nn.L1Loss(reduction = self.reduction)
        self.MSE_loss = torch.nn.MSELoss(reduction = self.reduction)

    def __call__(self, input_y, target):
        return self.L1_loss(input_y, target)*self.alpha + (1-self.alpha)*self.MSE_loss(input_y, target).item()


def gradient_reg(cp_detect, x, p=0.5):
    """
    To be applied with transition masking
    """
    assert cp_detect, "Cannot apply gradient regularization without transition masking"
    x_diff = get_gradient(x)   
    return torch.mean(torch.pow(torch.abs(x_diff), p)).item()
    #return torch.mean(p*torch.log(torch.abs(x_diff)))

class Changepoint_Metrics(object):
    def __init__(self, name, seq_len=317, win_tol=2):
        self.Precision =None
        self.Recall = None #same as A_cp
        self.Accuracy = None
        self.A_no_cp = None
        self.Balanced_Accuracy = None
        self.seq_len = seq_len
        self.win_tol = win_tol
        self.name = name
        
    def __call__(self, cp_target, cp_pred):
        self.Precision, self.Recall, self.Accuracy, self.A_no_cp, self.Balanced_Accuracy = \
        eval_cp_batch(cp_target, cp_pred, self.seq_len, self.win_tol)
    
    def get_name(self):
        print(self.name)

class Changepoint_gradient(Changepoint_Metrics):
    def transform(self, cp_pred_raw):
        Batch_size, T = cp_pred_raw.shape[0], cp_pred_raw.shape[1]
        gradient_thresh = 0.18
        cp_gradient_pred = torch.zeros((Batch_size, T))
        cp_gradient_idx = torch.nonzero(torch.abs(cp_pred_raw[:,1:,:]-cp_pred_raw[:,:-1,:])>gradient_thresh)
        cp_gradient_pred[cp_gradient_idx[:,0], cp_gradient_idx[:,1]]=1
        return cp_gradient_pred
    
class Changepoint_mc_drop(Changepoint_Metrics):
    def transform(self, cp_pred_raw):
        Batch_size, T = cp_pred_raw.shape[0], cp_pred_raw.shape[1]
        mc_dropout_thresh = 0.10
        cp_mc_pred = torch.zeros((Batch_size, T))
        cp_mc_idx = torch.nonzero(cp_pred_raw>mc_dropout_thresh)
        cp_mc_pred[cp_mc_idx[:,0], cp_mc_idx[:,1]]=1
        return cp_mc_pred
