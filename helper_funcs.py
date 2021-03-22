import pickle
import numpy as np
import pandas as pd
import torch
import json
import logging
import os
import shutil
import os.path as osp
import allel

def weight_int(m):
    if isinstance(m, torch.nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                 torch.nn.init.xavier_normal_(param)
            
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
    
class Params():
    """Class that loads hyperparameters from a json file
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
def load_path(path, en_pickle=False, en_df=False):
    with open (path, 'rb') as f:
        if en_pickle:
            file_content = pickle.load(f)
        elif en_df:
            file_content = pd.read_csv(f, sep="\t", header=None)
        else:
            file_content = np.load(f)
    return file_content

def save_file(path, file, en_pickle=False, en_df=False ):
    with open (path, 'wb') as f:
        if en_pickle:
            pickle.dump(file, f)
        elif en_df:
            file.to_csv(path, sep="\t", index=False)
        else:
            np.save(f, file)
            
def save_checkpoint(state, save_path, is_best):
    
    if not osp.exists(save_path):
        print(f'Checkpoint path does not exists, making {save_path}')
        os.mkdir(save_path)
        
    checkpoint = osp.join(save_path, 'last.pt')
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, osp.join(save_path, 'best.pt'))
    
def load_model(model_path, model_ls, optimizer=None):
    if not osp.exists(model_path):
        # ToDo look into the raise exception error not
        # coming from BaseException
        print(f'{model_path} does not exist')
        raise (f'{model_path} does not exist')
        
    checkpoint = torch.load(model_path)

    print(f"best val accuracy : {checkpoint['accr']['accr']}")
    print(f"at epoch : {checkpoint['epoch']}")
    print(f"train accuracy: {checkpoint['train_accr']['accr']}")
    
    for i, model_state in enumerate(checkpoint['model_state_dict']):
        model_ls[i].load_state_dict(model_state)
         
    return model_ls

def early_stopping(val_this_accr, val_prev_accr, patience, thresh):
    """
    If for consecutively, acc does not decrease more than thresh or increases, then stop 
    """
    if ((val_this_accr - val_prev_accr) > thresh) or (abs(val_this_accr - val_prev_accr) <= thresh):
        patience += 1
    else:
        patience = 0
    
    return patience

def set_logger(log_path):
    """Set the logger to log info in terminal and file log_path
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not osp.exists(log_path):
        print(f'Logging path does not exist, making {log_path}')
        os.mkdir(log_path)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path + ".log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

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
    mean = mat_vcf_np.sum(axis=0, keepdims=True)/mat_vcf_np.shape[0]
    std = np.sqrt(np.mean(np.absolute(mat_vcf_np-mean)**2, axis=0, keepdims=True))
    
    # filter snps to informative snps -- if std > threshold
    filtered_snp_idx = np.where(std[0,:]>=filter_thresh)[0]
    return mean[0,filtered_snp_idx], std[0,filtered_snp_idx], filtered_snp_idx

def convert_nVector(coord_map):
    lat, lon = coord_map
    lat *= np.pi / 180
    lon *= np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    nVector = [x, y, z]
    return nVector

def convert_coordinates(nVector):
    xcord, ycord, zcord = nVector
    coord_x = np.arctan2(zcord, np.hypot(xcord, ycord)) * 180 / np.pi
    coord_y = np.arctan2(ycord, xcord) * 180 / np.pi

    return coord_x, coord_y

def square_normalize(y_pred):
    """
    square normalize a tensor with 3 dimensions - x, y and z
    used for normalizing for Geographical n vector
    """
    y_pred_square = torch.pow(y_pred, 2)
    temp = torch.sum(y_pred_square, dim=2).reshape(y_pred_square.shape[0], y_pred_square.shape[1], 1)
    y_pred_transformed = y_pred / torch.sqrt(temp)
    return y_pred_transformed

def split_batch(seq_batch, bptt):
    """
    Split torch.tensor batch by bptt steps,
    Split sequence dim by bptt
    """
    batch_splits = seq_batch.split(bptt, dim=1)
    return batch_splits
    
def activate_mc_dropout(*models):
    for model in models:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                

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
    plt.plot(var_sorted)
    plt.show()

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

def interpolate_genetic_pos(df_gm_pos, df_gm_chm):
    xvals = df_gm_pos['physical_pos']
    yinterp = np.interp(xvals, df_gm_chm['physical_pos'], df_gm_chm['genetic_pos'])
    df_gm_pos['genetic_pos'] = yinterp
    return df_gm_pos
    
def form_windows(df_snp_pos, params):
    recomb_w = torch.tensor(df_snp_pos['genetic_pos'].values[0:params.chmlen]\
        .reshape(params.n_win, params.win_size)).float()
    recomb_w = torch.max(recomb_w, dim=1)[0]
    return recomb_w

def get_gradient(x):
    x_diff = x[:,:-1,:]-x[:,1:,:]
    #insert 0 at the end
    x_diff = torch.cat((x_diff, torch.zeros_like(x[:,0,:]).unsqueeze(1)), dim=1)
    return x_diff

class Running_Average():
    def __init__(self, num_members):
        self.num_members = num_members
        if self.num_members > 1:
            self.value = [0]*num_members
            self.steps = [0]*num_members
        else:
            self.value = 0
            self.steps = 0
    
    def update(self, val, step_size):
        if self.num_members > 1:
            self.value = [sum(x) for x in zip(val, self.value)]
            self.steps = [sum(x) for x in zip(step_size, self.steps)]
        else:
            self.value += val
            self.steps += step_size
        
    def __call__(self):
        if self.num_members > 1:
            return [x/float(y) for x,y in zip(self.value, self.steps)]
        else:
            return self.value/float(self.steps)
        
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
        
# function for custom learning rate
class custom_opt():
    def __init__(self, optimizer, d_model, warmup_steps, factor, groups=[2,3]):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step_num = 0
        self._rate = 0
        self.factor = factor
        self.groups = groups

    def step(self):
        self._step_num += 1
        rate = self.rate()
        for i in self.groups:
            self.optimizer.param_groups[i]['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step_num=None):
        if step_num==None:
            step_num=self._step_num
        return self.factor * (self.d_model**(-0.5)*min(step_num**(-0.5),\
                                                       step_num*(self.warmup_steps**(-1.5))))