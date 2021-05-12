import numpy as np
import torch
from src.utils.dataUtil import get_gradient
from collections import namedtuple

t_cp_accr = namedtuple('t_cp_accr', ['loss_cp', 'Precision', 'Recall', 'BalancedAccuracy'])
t_cp_accr.__new__.__defaults__=(None,)*len(t_cp_accr._fields)
t_sp_accr = namedtuple('t_sp_accr', ['loss_sp', 'Precision', 'Recall', 'BalancedAccuracy'])
t_sp_accr.__new__.__defaults__=(None,)*len(t_sp_accr._fields)
t_accr = namedtuple('t_accr', ['l1_loss', 'mse_loss', 'smoothl1_loss', 'weighted_loss',\
    'loss_main', 'loss_aux', 'residual_loss', 'gcdLoss', 'accAtGcd'])
t_accr.__new__.__defaults__=(None,)*len(t_accr._fields)
t_out = namedtuple('t_out', ['coord_aux','coord_main', 'cp_logits', 'y_var', 'sp'])
t_out.__new__.__defaults__=(None,)*len(t_out._fields)
t_balanced_gcd = namedtuple('t_balanced_gcd', ['median_gcd', 'meanBalancedGcdSp', \
    'meanBalancedGcdGp', 'medianBalancedGcdSp', 'medianBalancedGcdGp'])
t_balanced_gcd.__new__.__defaults__=(None,)*len(t_out._fields)
t_results = namedtuple('t_results',['t_accr', 't_cp_accr', 't_sp_accr', 't_out', 't_balanced_gcd'])
t_results.__new__.__defaults__=(None,)*len(t_results._fields)


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
            return z.sum()
        return z.mean()
    
class Weighted_Loss():
    def __init__(self, reduction='mean', alpha = 1.0):
        self.alpha = alpha
        self.reduction = reduction
        self.L1_loss = torch.nn.L1Loss(reduction = self.reduction)
        self.MSE_loss = torch.nn.MSELoss(reduction = self.reduction)

    def __call__(self, input_y, target):
        return self.L1_loss(input_y, target)*self.alpha + (1-self.alpha)*self.MSE_loss(input_y, target)

def gradient_reg(cp_detect, x, p=0.5):
    """
    To be applied with transition masking
    """
    assert cp_detect, "Cannot apply gradient regularization without transition masking"
    x_diff = get_gradient(x)   
    return torch.mean(torch.pow(torch.abs(x_diff), p))
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

class GcdLoss():
    def __init__(self):
        self.eps=1e-4
        self.earth_radius=6371
        self._batchGcdLs=[]
        self.__classGcdSuperpop, self.__classGcdGranularpop={}, {}
        
    def _rawGcd(self, input_y, target):
        return torch.acos(torch.sum(input_y * target, dim=2).clamp(-1.0 + self.eps, 1.0 - self.eps))

    def __call__(self, input_y, target):
        """
        returns sum of gcd given prediction label input_y of shape (n_samples x n_windows)
        and target label target of shape (n_sampled x n_windows)
        """
        rawGcd = self._rawGcd(input_y, target)
        sum_gcd = torch.sum(rawGcd) * self.earth_radius
        return sum_gcd

    def median(self):
        return torch.median(torch.cat(self._batchGcdLs)).item() 

    def balancedGcd(self, superpop, granular_pop):
        gcdTensor = self._batchGcdLs[-1]
        superpop_num=np.unique(superpop)
        granularpop_num=np.unique(granular_pop)
        for i in superpop_num:
            if self.__classGcdSuperpop.get(i) is None: self.__classGcdSuperpop[i]=Running_Average()
            idx=torch.nonzero(superpop==i)
            self.__classGcdSuperpop[i].update(gcdTensor[idx[:,0], idx[:,1]].sum(), len(idx))
            
        for i in granularpop_num:
            if self.__classGcdGranularpop.get(i) is None: self.__classGcdGranularpop[i]=Running_Average()
            idx=torch.nonzero(granular_pop==i)
            self.__classGcdGranularpop[i].update(gcdTensor[idx[:,0], idx[:,1]].sum(), len(idx))
            
    def meanBalanced(self):
        meanBalancedSuperpop=torch.mean(torch.tensor([self.__classGcdSuperpop[k]() for k in self.__classGcdSuperpop.keys()]))
        meanBalancedGranularpop=torch.mean(torch.tensor([self.__classGcdGranularpop[k]() for k in self.__classGcdGranularpop.keys()]))
        return meanBalancedSuperpop.item(), meanBalancedGranularpop.item()

    def medianBalanced(self):
        medianBalancedSuperpop=torch.median(torch.tensor([self.__classGcdSuperpop[k]() for k in self.__classGcdSuperpop.keys()]))
        medianBalancedGranularpop=torch.median(torch.tensor([self.__classGcdGranularpop[k]() for k in self.__classGcdGranularpop.keys()]))
        return medianBalancedSuperpop.item(), medianBalancedGranularpop.item()

    def accAtGcd(self, input_y, target, gcdThresh):
        batchGcd = self._rawGcd(input_y, target).detach() * self.earth_radius
        self._batchGcdLs.append(batchGcd)
        accAtGcd = len(batchGcd[batchGcd<=gcdThresh])
        return accAtGcd

def class_accuracy(y_pred, y_test):
    correct_pred = (y_pred == y_test).astype(float)
    n = y_test.shape[0]
    w = y_test.shape[1]
    acc = correct_pred.sum() / (n * w)
    acc = acc * 100
    return acc

class Running_Average():
    def __init__(self):
        self.value = 0
        self.steps = 0
    
    def update(self, val, step_size):
        self.value += val
        self.steps += step_size
        
    def __call__(self):
        return self.value/float(self.steps)


