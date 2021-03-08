import numpy as np
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
from helper_funcs import load_path
import os.path as osp

class Haplotype(Dataset):
    def __init__(self, dataset_type, path_prefix, params, labels_path):
        if dataset_type not in ["train", "valid", "test"]:
            raise ValueError

        if dataset_type=="train":
            self.gens_to_ret =  params.train_gens
        elif dataset_type=="valid":
            self.gens_to_ret =  params.valid_gens
        else:
            self.gens_to_ret =  params.test_gens
        
        print(f" Loading {dataset_type} Dataset")
        
        for i, gen in enumerate(self.gens_to_ret):
            print(f"Loading gen {gen}")
            curr_snps = load_path(osp.join(path_prefix, str(dataset_type) ,'gen_' + str(gen), 'mat_vcf_2d.npy'))
            print(f' snps data: {curr_snps.shape}')
            curr_vcf_idx = load_path(osp.join(path_prefix , str(dataset_type) ,'gen_' + str(gen) ,'mat_map.npy'))
            print(f' y_labels data :{curr_vcf_idx.shape}')

            if i>0:
                self.snps = np.concatenate((self.snps, curr_snps),axis=0)
                self.vcf_idx = np.concatenate((self.vcf_idx, curr_vcf_idx),axis=0)
            else:
                self.snps = curr_snps
                self.vcf_idx = curr_vcf_idx

        self.coordinates = load_path(osp.join(labels_path, params.coordinates), en_pickle=True)
        self.data = {'X':None, 'y':None, 'cp_mask':None}
        self.load_data(params)
    
    def __len__(self):
        return len(self.data['X']) 
    
    def mapping_func(self, arr, b, dim):
        """
        Inputs:
        arr: 3D array
        b : dict with 3 dim array as values
        return:
        result: 3D array
        """
        result = np.zeros((arr.shape[0], arr.shape[1], dim)).astype(float)
        
        for k in np.unique(arr):
            idx = np.nonzero(arr==k)
            for d in np.arange(dim):
                result[idx[0], idx[1], d]=b[k][d]
            
        result = torch.tensor(result).float()
        return result
    
    def load_data(self, params):        
        # take the mode according to windows for labels
        # map to coordinates according to ref_idx
        print("Transforming the data")
        self.data['X'] = torch.tensor(self.snps[:,0:params.chmlen])
        y_tmp = torch.tensor(self.vcf_idx[:,0:params.chmlen])
        y_tmp = y_tmp.reshape(-1, params.n_win, params.win_size)
        y_vcf_idx = torch.mode(y_tmp, dim=2)[0]
        
        self.data['y'] = self.mapping_func(y_vcf_idx.detach().cpu().numpy(), self.coordinates, params.dataset_dim)
        
        if params.cp_detect:
            # if the only gen is gen 0 then there will be no changepoints with founders
            assert max(self.gens_to_ret)>0, "No changepoints will be found. Feeding only founders."
            cps = self.data['y'][:,:-1,:]-self.data['y'][:,1:,:]
            #insert 0 at the end
            cps = torch.cat((cps, torch.zeros_like(self.data['y'][:,0,:]).unsqueeze(1)), dim=1)
            
            assert cps.sum()!=0, "No changepoints found. Check the input file"
            cp_idx = torch.nonzero(cps)[0]
            for i in cp_idx:
                cps[i-params.cp_tol:i+params.cp_tol] = cps[i]
            #revert to make a mask
            self.data['cp_mask'] = torch.where(cps==0.0, torch.tensor([1.0]), torch.tensor([0.0]))
        else:
            self.data['cp_mask'] = torch.ones_like(self.data['y'])
    
    def __getitem__(self, idx):
        ls =[]
        for k, v in self.data.items():
            if v is not None:
                ls.append(self.data[k][idx])
        return ls

