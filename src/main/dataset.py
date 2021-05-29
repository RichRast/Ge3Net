import numpy as np
import pandas as pd 
import torch 
from torch.utils.data import Dataset
import logging
import math
import os.path as osp
from src.utils.dataUtil import load_path, getWinInfo
from src.utils.labelUtil import repeat_pop_arr
from src.utils.modelUtil import convert_nVector
from src.utils.decorators import timer
import pdb
import snoop

class Haplotype(Dataset):
    @timer
    def __init__(self, dataset_type, params, data_dir, **kwargs):
        labels_path = kwargs.get('labels_path')
        if dataset_type not in ["train", "valid", "test", "no_label"]:
            raise ValueError
        
        self.params = params
        if dataset_type=="train":
            self.gens_to_ret =  self.params.train_gens
        elif dataset_type=="valid":
            self.gens_to_ret =  self.params.valid_gens
        elif dataset_type=="test":
            self.gens_to_ret =  self.params.test_gens
        
        logging.info(f" Loading {dataset_type} Dataset")

        # can add more here, example granular_pop is not being used
        self.data = {'X':None, 'y':None, 'y_vcf_idx':None, 'cps':None, 'superpop':None, 'granular_pop':None}
        
        if labels_path is None:
            logging.info(f'Loading snps data')
            self.snps = load_path(osp.join(data_dir, str(dataset_type),'mat_vcf_2d.npy'))
            logging.info(f"snps data shape : {self.data['X'].shape}")
        else:
            for i, gen in enumerate(self.gens_to_ret):
                logging.info(f"Loading gen {gen}")
                curr_snps = load_path(osp.join(data_dir, str(dataset_type) ,'gen_' + str(gen), 'mat_vcf_2d.npy'))
                logging.info(f' snps data: {curr_snps.shape}')
                curr_vcf_idx = load_path(osp.join(labels_path , str(dataset_type) ,'gen_' + str(gen) ,'mat_map.npy'))
                logging.info(f' y_labels data :{curr_vcf_idx.shape}')

                if i>0:
                    self.snps = np.concatenate((self.snps, curr_snps),axis=0)
                    self.vcf_idx = np.concatenate((self.vcf_idx, curr_vcf_idx),axis=0)
                else:
                    self.snps = curr_snps
                    self.vcf_idx = curr_vcf_idx
   
        chmlen, n_win = getWinInfo(self.snps.shape[1], self.params.win_size)
        params.n_win = n_win
        params.chmlen = chmlen
        self.data['X'] = torch.tensor(self.snps[:,:chmlen]).float()

        if labels_path is not None:
            pop_sample_map = pd.read_csv(osp.join(labels_path, self.params.pop_sample_map), sep='\t')
            self.pop_arr = repeat_pop_arr(pop_sample_map)
            self.coordinates = load_path(osp.join(labels_path, self.params.coordinates), en_pickle=True)
            self.transform_data(chmlen, n_win)
        
    def __len__(self):
        return len(self.data['X']) 
    
    def mapping_func(self, arr, b, dim):
        """
        Inputs:
        arr: 2(d)D array
        b : dict with 3(d) dim array as values
        d: dimension of the output, could be 3 or more
        return:
        result: 3(d)D array
        """
        if self.params.geography:
            dim=2
        result = np.zeros((arr.shape[0], arr.shape[1], dim)).astype(float)
        
        for k in np.unique(arr):
            idx = np.nonzero(arr==k)
            for d in np.arange(dim):
                result[idx[0], idx[1], d]=b[k][d]
            
        if self.params.geography:
            result=self._geoConvertLatLong2nVec(result)
        result = torch.tensor(result).float()
        return result

    def _geoConvertLatLong2nVec(self, coord):
        """
        Converts the result from 2 dim Lat/Long to 3 dim n vector
        """
        # ToDo: Need to change this. Too slow!!!
        lat=coord[..., 0]
        long=coord[..., 1] 
        nVec=convert_nVector(lat,long)
        return nVec

    def pop_mapping(self, y_vcf, pop_arr, type='superpop'):
        
        result = np.zeros((y_vcf.shape[0], y_vcf.shape[1])).astype(float)
        if type=='superpop':
            col_num=3
        elif type=='granular_pop':
            col_num=2
        for k in np.unique(y_vcf):
            idx = np.nonzero(y_vcf==k)
            pop_arr_idx = np.nonzero(pop_arr[:,1]==k)[0]
            result[idx[0], idx[1]]=pop_arr[pop_arr_idx, col_num]
        result = torch.tensor(result).float()
        return result

    @timer
    def transform_data(self, chmlen, n_win):        
        # take the mode according to windows for labels
        # map to coordinates according to ref_idx
        logging.info("Transforming the data")
        y_tmp = torch.tensor(self.vcf_idx[:,:chmlen])
        y_tmp = y_tmp.reshape(-1, n_win, self.params.win_size)
        self.data['y_vcf_idx'] = (torch.mode(y_tmp, dim=2)[0]).detach().cpu().numpy()
        
        self.data['y'] = self.mapping_func(self.data['y_vcf_idx'], self.coordinates, self.params.dataset_dim)
        self.data['superpop'] = self.pop_mapping(self.data['y_vcf_idx'], self.pop_arr, type='superpop')
        self.data['granular_pop'] = self.pop_mapping(self.data['y_vcf_idx'], self.pop_arr, type='granular_pop')
        
        if self.params.cp_detect:
            # if the only gen is gen 0 then there will be no changepoints with founders
            assert max(self.gens_to_ret)>0, "No changepoints will be found. Feeding only founders."
            cps = self.data['granular_pop'][:,:-1]-self.data['granular_pop'][:,1:] #window dim is 1 less
            assert cps.sum()!=0, "No changepoints found. Check the input file"
            # find window indices where diff for any dim !=0
            cps_copy = torch.zeros_like(self.data['granular_pop'], dtype=torch.uint8)
            cps_idx = torch.nonzero(cps)
            cps_copy[cps_idx[:,0], cps_idx[:,1]] = 1
            for i in range(1,self.params.cp_tol+1):
                tolVal=math.ceil(i/2)
                if i%2==1:
                    cpsWin = list(map(lambda x: min(x+tolVal, cps.shape[1]), cps_idx[:,1]))
                else:
                    cpsWin = list(map(lambda x: max(x-tolVal, 0), cps_idx[:,1]))
                cps_copy[cps_idx[:,0], cpsWin] = 1
            self.data['cps'] = cps_copy
            del cps, cps_copy
        else:
            self.data['cps'] = torch.zeros_like(self.data['granular_pop'], dtype=torch.uint8)
        
        torch.cuda.empty_cache()
    
    def __getitem__(self, idx):
        ls =[]
        for k, v in self.data.items():
            ls.append(self.data[k][idx])
        return ls


