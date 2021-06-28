import pickle
import numpy as np
import pandas as pd
import torch
import json
import logging
import os
import os.path as osp
import allel
import copy
from typing import Tuple, List
from src.utils.decorators import guardAgainstDivideZero, timer
    
def load_path(path, en_pickle=False, en_df=False):
    
    with open (path, 'rb') as f:
        try:
            if en_pickle:
                file_content = pickle.load(f)
            elif en_df:
                file_content = pd.read_csv(f, sep="\t", header=None)
            else:
                file_content = np.load(f)
            
        except FileNotFoundError as fnfe:
            logging.exception(fnfe)
        except Exception as e:
            raise e
        return file_content
        
def save_file(path, file, en_pickle=False, en_df=False ):
    with open (path, 'wb') as f:
        if en_pickle:
            pickle.dump(file, f)
        elif en_df:
            file.to_csv(path, sep="\t", index=False)
        else:
            np.save(f, file)
       
def set_logger(log_path):
    """Set the logger to log info in terminal and file log_path
    """
    logger = logging.getLogger("Ge3Net")
    logger.setLevel(logging.DEBUG)
    if not osp.exists(log_path):
        print(f'Logging path does not exist, making {log_path}')
        os.makedirs(log_path)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path + ".log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(funcName)s:%(name)s:%(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def vcf2npy(vcf_file):
    """
    Reads vcf file and returns numpy with 
    both maternal and paternal snp data
    """
    vcf_data = allel.read_vcf(vcf_file)
    chm_len, nout, _ = vcf_data["calldata/GT"].shape
    mat_vcf_2d = vcf_data["calldata/GT"].reshape(chm_len,nout*2).T
    return mat_vcf_2d.astype('int16')

def filter_snps(mat_vcf_np, filter_thresh):
    """
    filters snps by a given threshold
    filter_thresh: variance filter
    """
    mean = divide(mat_vcf_np.sum(axis=0, keepdims=True),mat_vcf_np.shape[0])
    var_tmp = np.mean(np.absolute(mat_vcf_np-mean)**2, axis=0, keepdims=True)
    print(f"var_tmp max, min:{max(var_tmp)},{min(var_tmp)}")
    var = mat_vcf_np.var(axis=0, keepdims=True)
    print(f" variance shape, max and min var :{var.shape}, {max(var)}, {min(var)}")
    # filter snps to informative snps -- if std > threshold
    # filtered_snp_idx = np.where(std[0,:]>=filter_thresh)[0]
    filtered_snp_idx = np.where(var[0,:]>=filter_thresh)[0]
    return mean[0,filtered_snp_idx], var[0,filtered_snp_idx], filtered_snp_idx

def square_normalize(y_pred):
    """
    square normalize a tensor with 3 dimensions - x, y and z
    used for normalizing for Geographical n vector
    >>> square_normalize(torch.Tensor([[[1.0,3.0,5.0],[1.0,3.0,2.0]]]))
    tensor([[[0.1690, 0.5071, 0.8452],
             [0.2673, 0.8018, 0.5345]]])
    """
    eps=1e-4
    y_pred_square = torch.pow(y_pred, 2)
    tmp = torch.sum(y_pred_square, dim=2).reshape(y_pred_square.shape[0], y_pred_square.shape[1], 1)
    # square of each comp x, y and z of n vector must be positive but less than 1
    # clamp it with eps so as to avoid divide by zero error
    tmp = torch.clamp(tmp, min=eps)
    assert torch.any(tmp>eps), "one of the elements was below epsilon"
    y_pred_transformed = y_pred / torch.sqrt(tmp)
    return y_pred_transformed
                
def filter_vcf(vcf_filepath, thresh, verbose=True):
    """
    Return vcf file (dict) after subsetting for snps that have 
    variance > a given threshold
    """
    vcf_original = allel.read_vcf(vcf_filepath)
    samples = len(vcf_original['samples'])
    mat_vcf_2d = vcf2npy(vcf_filepath)
    original_snps = mat_vcf_2d.shape[1]
    
    if verbose:
        print(f'original vcf contains {original_snps} snps and {samples} samples ')

    var_vcf_original = mat_vcf_2d.var(axis=0)
    var_sorted = copy.deepcopy(var_vcf_original)
    var_sorted = np.sort(var_sorted)
    # plt.plot(var_sorted)
    # plt.show()

    var_thresh = var_sorted[var_sorted>thresh]
    idx_chosen = np.argwhere(np.isin(var_vcf_original, var_thresh))

    #subset the vcf file
    subsetted_vcf = {}
    for k,v in vcf_original.items():
        if v.shape[0] == original_snps:
            subsetted_vcf[k] = v[idx_chosen].squeeze(1)
        else:
            subsetted_vcf[k] = v
    if verbose:  
        print('original vcf shapes: \n')
        for k,v in vcf_original.items():
            print(f'{k} shape:{v.shape}')
        print('\n subsetted vcf shapes: \n')    
        for k,v in subsetted_vcf.items():
            print(f'{k} shape:{v.shape}')         
        print(f"new subsetted vcf contains {len(subsetted_vcf['variants/POS'])} \
        snps and {len(subsetted_vcf['samples'])} samples")

    return subsetted_vcf

@timer
def get_recomb_rate(genetic_map_path, vcf_file_path, chm='chr22'):   
    df_gm = load_path(genetic_map_path, en_df=True)
    df_gm.rename(columns={0:'chm_no.', 1:'physical_pos', 2:'genetic_pos'}, inplace=True)
    df_gm_chm = df_gm[df_gm['chm_no.']==chm].reset_index(drop=True)
    df_vcf = allel.read_vcf(vcf_file_path)
    
    # get the range of snps (example : 0 to 317410 for chm22)
    df_vcf_pos = pd.DataFrame(df_vcf['variants/POS'],columns={'physical_pos'})
    df_gm_pos = df_vcf_pos.merge(df_gm_chm[['physical_pos', 'genetic_pos']], how="left", \
                                      on='physical_pos')
    
    return df_gm_chm, df_vcf, df_gm_pos

@timer
def interpolate_genetic_pos(df_gm_pos, df_gm_chm):
    xvals = df_gm_pos['physical_pos']
    yinterp = np.interp(xvals, df_gm_chm['physical_pos'], df_gm_chm['genetic_pos'])
    df_gm_pos['genetic_pos'] = yinterp
    return df_gm_pos

@timer    
def form_windows(df_snp_pos, chmlen, win_size):
    recomb_w = torch.tensor(df_snp_pos['genetic_pos'].values[:chmlen]\
        .reshape(-1, win_size)).float()
    recomb_w = torch.max(recomb_w, dim=1)[0]
    return recomb_w

def get_gradient(x):
    x_diff = x[:,:-1,:]-x[:,1:,:]
    #insert 0 at the end
    x_diff = torch.cat((x_diff, torch.zeros_like(x[:,0,:]).unsqueeze(1)), dim=1)
    return x_diff
        
def form_mask(input_tensor, device):
    """
    return a mask for majority class=1
    to be used for masking loss
    """
    input_x=input_tensor.detach().clone()
    input_x=input_x.bool()
    
    #form a matrix where all elements are zeros except randomly
    #samples by each row a percentage masked_perc
    weight=torch.rand(input_x.shape).to(device)
    sampled_tensor = weight.round().bool()

    mask = (input_x & sampled_tensor).int()

    return mask
        
def getValueBySelection(val1, **kwargs):
    """
    This function returns the value of col2 for the row
    where col1 value matches val1 for a given 2d array arr
    """
    arr=kwargs.get("arr")
    col1=kwargs.get("col1")
    col2=kwargs.get("col2")
    return arr[np.where(arr[:,col1]==val1)[0],col2]
    
def getWinInfo(chmLen: int, winSize:int)->Tuple[int, int]:
    """
    Given the raw chm length and the window size, truncate the 
    last window snps and return the truncated chm length and
    number of windows
    """
    nWin = int(chmLen/winSize)
    newChmLen = winSize*nWin
    return newChmLen, nWin

@guardAgainstDivideZero
def divide(a,b):
    return a/b

def getFstIndex(pop_dict: dict, vcf_file: np.array, **kwargs)->np.array:
    """
    computes Fst index according to definition in 
    https://en.wikipedia.org/wiki/Fixation_index
    """
    winSize=kwargs.get("winSize")
    ...

def getHudsonFst(pop_dict: dict, vcf_file: np.array)->np.array:
    """
    computes Fst index according to equation 3 in 
    https://en.wikipedia.org/wiki/Fixation_index

    """
    ...
