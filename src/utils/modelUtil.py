import torch
import yaml
import os
import shutil
import os.path as osp
import numpy as np
from torch import nn
import torch.nn.functional as F
from src.utils.decorators import timer


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
    """Class that loads hyperparameters from a yaml file
    """
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
            self.__dict__.update(params)

    def save(self, yaml_path):
        # for a custom class or an object such as device(type=cuda),
        # provide a string representation, like the default=str argument for 
        # json dump
        yaml.SafeDumper.yaml_representers[None] = lambda self, data: \
        yaml.representer.SafeRepresenter.represent_str(self, str(data))
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(self.__dict__, f, indent=4)

    def update(self, newParams_dict):
        """updates new parameters to the existing params dict"""
        self.__dict__.update(newParams_dict)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
def save_checkpoint(state, save_path, is_best):

    if not osp.exists(save_path):
        print(f'Checkpoint path does not exists, making {save_path}')
        os.makedirs(save_path)
        
    checkpoint = osp.join(save_path, 'last.pt')
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, osp.join(save_path, 'best.pt'))
    
@timer
def load_model(model_weights_path, model_init, **kwargs):
    device_gpu=kwargs.get('device_gpu')
    print(f"device in load model:{device_gpu}")
    if not osp.exists(model_weights_path):
        # ToDo look into the raise exception error not
        # coming from BaseException
        print(f'{model_weights_path} does not exist')
        raise (f'{model_weights_path} does not exist')
    if not device_gpu:
        checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_weights_path)
    model_init.load_state_dict(checkpoint['model_state_dict'])        
    return model_init

def loadTrainingStats(trainingStats_path, **kwargs):
    device_gpu=kwargs.get('device_gpu')
    if not osp.exists(trainingStats_path):
        # ToDo look into the raise exception error not
        # coming from BaseException
        print(f'{trainingStats_path} does not exist')
        raise (f'{trainingStats_path} does not exist')
    
    if not device_gpu:
        checkpoint = torch.load(trainingStats_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(trainingStats_path)

    print(f"best val loss metrics : {checkpoint['val_accr']['t_accr']}")
    # print(f"test loss metrics : {checkpoint['test_accr']['t_accr']}")
    print(f"at epoch : {checkpoint['epoch']}")
    print(f"train loss metrics: {checkpoint['train_accr']['t_accr']}")

    print(f"best val cp metrics : {checkpoint['val_accr']['t_cp_accr']}")
    # print(f"test cp metrics : {checkpoint['test_accr']['t_cp_accr']}")
    print(f"train cp metrics: {checkpoint['train_accr']['t_cp_accr']}")

    print(f"best val sp metrics : {checkpoint['val_accr']['t_sp_accr']}")
    # print(f"test sp metrics : {checkpoint['test_accr']['t_sp_accr']}")
    print(f"train sp metrics: {checkpoint['train_accr']['t_sp_accr']}")

    print(f"best val balanced gcd metrics : {checkpoint['val_accr']['t_balanced_gcd']}")
    # print(f"test balanced gcd metrics : {checkpoint['test_accr']['t_balanced_gcd']}")
    print(f"train balanced gcd metrics: {checkpoint['train_accr']['t_balanced_gcd']}")
    
    model_stats=checkpoint['val_accr']
         
    return model_stats

def early_stopping(val_this_accr, val_prev_accr, patience, thresh):
    """
    If for consecutively, acc does not decrease more than thresh or increases, then stop 
    """
    if ((val_this_accr - val_prev_accr) > thresh) or (abs(val_this_accr - val_prev_accr) <= thresh):
        patience += 1
    else:
        patience = 0
    
    return patience

def convert_nVector(lat, lon):
    """
    Used for a batch of numpy matrix
    >>> convert_nVector(np.array([[54.00366 ,64.49884603]]), np.array([[-2.547855, 26.2746656]]))

    """
    # lat, lon = coord_map
    lat *= np.pi / 180
    lon *= np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    x=x[...,np.newaxis]
    y=y[...,np.newaxis]
    z=z[...,np.newaxis]
    nVector = np.stack((x, y, z), axis=-2).squeeze(-1)
    return nVector

def convert_coordinates(*nVector):
    """
    >>> convert_coordinates()
    """
    xcord, ycord, zcord = nVector
    coord_x = np.arctan2(zcord, np.hypot(xcord, ycord)) * 180 / np.pi
    coord_y = np.arctan2(ycord, xcord) * 180 / np.pi
    coord_x=coord_x[...,np.newaxis]
    coord_y=coord_y[...,np.newaxis]
    coordinates=np.stack((coord_x, coord_y), axis=-2).squeeze(-1)
    return coordinates

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

class custom_opt():
    """
    class for custom learning rate
    """
    def __init__(self, optimizer, d_model, warmup_steps, factor, groups=[2,3]):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step_num = 0
        self._rate = 0
        self.factor = factor
        self.groups = groups

    def state_dict(self):
        return self.optimizer.state_dict()

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
                
class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__attr__(name)
        except AttributeError:
            return getattr(self.module, name)

def countParams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class Swish(nn.Module):
    def __init__(self, params):
        super(Swish, self).__init__()

    def forward(x):
        return x * F.sigmoid(x)

def modelAveraging(model_paths, model_init):
    """
    Given the path for model checkpoints, read the model weights
    avergage them and return the ensembled model
    """
    