import numpy as np
import pandas as pd
import torch
from src.utils.dataUtil import get_gradient
from collections import namedtuple
from dataclasses import dataclass
from typing import Any
from enum import Enum
import pdb

t_results = namedtuple('t_results',['t_accr', 't_cp_accr', 't_sp_accr', 't_out', 't_balanced_gcd'])
t_results.__new__.__defaults__=(None,)*len(t_results._fields)
t_prCounts = namedtuple('t_prCounts', ['TP', 'FP', 'TN', 'FN'])
t_prMetrics = namedtuple('t_prMetrics', ['Precision', 'Recall', 'Accuracy', 'A_major', 'BalancedAccuracy'])

cpMethod=Enum('cpMethod', 'neural_network gradient mc_dropout BOCD', start=0)

@dataclass
class modelOuts:
    coord_aux:Any=None
    coord_main:Any=None
    cp_logits:Any=None
    y_var:Any=None
    sp:Any=None

@dataclass
class RnnResults:
    out:Any=None
    out_nxt:Any=None
    loss_main:Any=None

@dataclass
class branchLoss:
    loss_main:Any=None
    loss_aux:Any=None
    loss_cp:Any=None

def eval_cp_matrix(true_cps, pred_cps, seq_len, win_tol=2):
    """
    For a given haplotype, compute the accuracy metrics for changepoint
    true_cp = np.array of true cp indices
    pred_cp = np.array of predicted cp indices
    win_tol = +/- tolerance for location and count of cp
    
    >>> eval_cp_matrix(true_cps=np.array([3,2]), pred_cps=np.array([4,1]), seq_len=317)
    (1.0, 1.0, 1.0, 1.0, 1.0, array([[1., 2.],
           [2., 1.]]))

    >>> eval_cp_matrix(true_cps=np.array([3,2]), pred_cps=np.array([4,1,5,6]),seq_len=317)
    (0.6666666666666666, 1.0, 0.9968454258675079, 0.9968253968253968, 0.9984126984126984, array([[1., 2., 3., 4.],
           [2., 1., 2., 3.]]))
    """

    FP_count = 0
    FN_count = 0
    n = len(true_cps)
    m = len(pred_cps)
    total_count = seq_len
    assert total_count!=0, "check input, sequence length in eval_cp is 0"
    # assert statement to make sure fill_value > win_tol
    distance_matrix = np.full((n,m), fill_value=np.inf)

    if n==m==0:
        TP_count, FP_count, FN_count  = 0, 0 , 0
    elif n==0:
        TP_count, FP_count, FN_count = 0, m, 0
        distance_matrix = pred_cps
    elif m==0:
        TP_count, FP_count, FN_count = 0, 0, n
        distance_matrix = true_cps
    else:
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
    # pdb.set_trace()
    return TP_count, FP_count, FN_count, TN_count, distance_matrix

def eval_cp_batch(cp_target, cp_pred, seq_len, win_tol=2):
    """
    cp_pred : [Batch_size X T]
    1 for cp and 0 for no cp

    cp_target : [Batch_size X T]
    1 for cp and 0 for no cp

    """
    #Todo need to batchify this function and eval_cp_matrix func
    # and move tp tensors
    num_samples = cp_pred.shape[0]
    countsArr=np.zeros((num_samples,4))
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
        
        countsArr[i,0], countsArr[i,1], countsArr[i,2], countsArr[i,3], _ = eval_cp_matrix(true_cps, pred_cps, seq_len, win_tol)
        # distance_matrix.append(distance_matrix_tmp)
    
    return t_prCounts(TP=countsArr[:,0], FP=countsArr[:,1], FN=countsArr[:,2], TN=countsArr[:,3])

def computePrMetric(prCounts):
    TP, FP, TN, FN=prCounts.TP, prCounts.FP, prCounts.TN, prCounts.FN
    
    total_count = TP+FP+TN+FN
    def getMetric(num, den):
        return np.divide(num, den, out = np.ones_like(num), where =den!=0)

    Precision = getMetric(num=TP, den=TP+FP)
    Recall = getMetric(num=TP, den=TP+FN)
    Accuracy = getMetric(num=TP+TN, den=total_count)
    A_major=getMetric(num=TN, den=TN+FP)
    BalancedAccuracy=0.5*(Recall+A_major)
    return t_prMetrics(Precision=Precision.sum(0), Recall=Recall.sum(0), Accuracy=Accuracy.sum(0), \
    A_major=A_major.sum(0), BalancedAccuracy=BalancedAccuracy.sum(0))._asdict()

class SmoothL1Loss():
    def __init__(self, reduction='mean', alpha=1.0):
        self.alpha = alpha
        self.reduction = reduction

    def __call__(self, input_y, target):
        mask=((input_y-target)<=self.alpha).float()
        rev_mask=((input_y-target)>self.alpha).float()
        
        z1 = mask*0.5*(torch.pow((input_y-target),2)/self.alpha)
        z2 = rev_mask*(abs(input_y - target) - 0.5*self.alpha)
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

def reportChangePointMetrics(name : str, cp_pred_raw: torch.Tensor, cp_target: torch.tensor, seq_len: int, cpThresh: float)->t_prMetrics:
    Batch_size, T = cp_pred_raw.shape[0], cp_pred_raw.shape[1]
    cp_pred = torch.zeros((Batch_size, T))
    if name==cpMethod.gradient.name:
        cp_idx = torch.nonzero(torch.abs(cp_pred_raw[:,1:,:]-cp_pred_raw[:,:-1,:])>cpThresh)
        cp_pred[cp_idx[:,0], cp_idx[:,1]]=1
    elif name==cpMethod.mc_dropout.name:
        cp_idx = torch.nonzero(cp_pred_raw>cpThresh)
        cp_pred[cp_idx[:,0], cp_idx[:,1]]=1
    elif name==cpMethod.neural_network.name:
        cp_pred = (torch.sigmoid(cp_pred_raw)>cpThresh).int()
    elif name==cpMethod.BOCD.name:
        cp_pred_diff = cp_pred_raw[:,1:]-cp_pred_raw[:,:-1]
        cp_pred = torch.zeros((Batch_size, T))
        cp_idx = torch.nonzero((cp_pred_diff<-cpThresh))
        for i,j in zip(cp_idx[:,0], cp_idx[:,1]):
            if j+1 < seq_len:
                if cp_pred_raw[i,j+1]<cpThresh:
                    cp_pred[i,j]=1
        
    prCounts = eval_cp_batch(cp_target, cp_pred, seq_len)
    prMetricsSum = computePrMetric(prCounts)
    prMetrics={}
    for k,v in prMetricsSum.items():
        prMetrics[k] = v/Batch_size
    return prMetrics, cp_pred

class Running_Average():
    def __init__(self):
        self.value = 0
        self.steps = 0
    
    def update(self, val, step_size):
        self.value += val
        self.steps += step_size
        
    def __call__(self):
        return self.value/float(self.steps)

class GcdLoss():
    def __init__(self):
        self.eps=1e-4
        self.earth_radius=6371
        self.gcdThresh=1000.0

    def rawGcd(self, input_y, target):
        return torch.acos(torch.sum(input_y * target, dim=2).clamp(-1.0 + self.eps, 1.0 - self.eps)) * self.earth_radius

    def __call__(self, input_y, target):
        """
        returns sum of gcd given prediction label input_y of shape (n_samples x n_windows)
        and target label target of shape (n_sampled x n_windows)
        """
        rawGcd = self.rawGcd(input_y, target)
        sum_gcd = torch.sum(rawGcd)
        return sum_gcd

@dataclass
class balancedMetrics():
    batchMetricLs=[]
    classSuperpop, classGranularpop={}, {}

    def fillData(self, dataTensor):
        self.batchMetricLs.append(dataTensor)

    def accAtThresh(self, gcdThresh=1000.0):
        batchGcd = self.batchMetricLs[-1]
        accAtGcd = len(batchGcd[batchGcd<=gcdThresh])
        return accAtGcd

    def medianMetric(self):
        return torch.median(torch.cat(self.batchMetricLs)).item() 

    def balancedMetric(self, superpop, granular_pop):
        gcdTensor = self.batchMetricLs[-1]
        superpop_num=np.unique(superpop).astype(int)
        granularpop_num=np.unique(granular_pop).astype(int)
        for i in superpop_num:
            if self.classSuperpop.get(i) is None: self.classSuperpop[i]=Running_Average()
            idx=torch.nonzero(superpop==i)
            self.classSuperpop[i].update(gcdTensor[idx[:,0], idx[:,1]].sum(), len(idx))
            
        for i in granularpop_num:
            if self.classGranularpop.get(i) is None: self.classGranularpop[i]=Running_Average()
            idx=torch.nonzero(granular_pop==i)
            self.classGranularpop[i].update(gcdTensor[idx[:,0], idx[:,1]].sum(), len(idx))             
                
    def meanBalanced(self):
        meanBalancedSuperpop=torch.mean(torch.tensor([self.classSuperpop[k]() for k in self.classSuperpop.keys()]))
        meanBalancedGranularpop=torch.mean(torch.tensor([self.classGranularpop[k]() for k in self.classGranularpop.keys()]))
        return meanBalancedSuperpop.item(), meanBalancedGranularpop.item()

    def medianBalanced(self):
        medianBalancedSuperpop=torch.median(torch.tensor([self.classSuperpop[k]() for k in self.classSuperpop.keys()]))
        medianBalancedGranularpop=torch.median(torch.tensor([self.classGranularpop[k]() for k in self.classGranularpop.keys()]))
        return medianBalancedSuperpop.item(), medianBalancedGranularpop.item()

def class_accuracy(y_pred, y_test):
    correct_pred = (y_pred == y_test).astype(float)
    n = y_test.shape[0]
    w = y_test.shape[1]
    acc = correct_pred.sum() / (n * w)
    acc = acc * 100
    return acc

def prMetricsByThresh(method_name, cp_pred_raw, cp_target, steps):
    num_samples = cp_target.shape[0]
    seqlen = cp_target.shape[1]
    min_prob = 0.0
    max_prob = 1.0
    increment = (max_prob - min_prob)/steps
    df=pd.DataFrame(columns=list(t_prMetrics._fields)+['thresh'])
    for thresh in np.arange(min_prob, max_prob + increment, increment):
        prMetrics = reportChangePointMetrics(method_name, cp_pred_raw, cp_target, seqlen, thresh)
        prMetrics['thresh']=thresh
        df=df.append(prMetrics, ignore_index=True)
    return df


