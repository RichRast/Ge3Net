import numpy as np
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
from helper_funcs import load_path
from build_labels_revised import repeat_pop_arr
import os.path as osp
from decorators import timer

class Haplotype(Dataset):
    def __init__(self, dataset_type, path_prefix, params, labels_path):
        if dataset_type not in ["train", "valid", "test", "no_label"]:
            raise ValueError
        
        self.params = params

        if dataset_type=="train":
            self.gens_to_ret =  self.params.train_gens
        elif dataset_type=="valid":
            self.gens_to_ret =  self.params.valid_gens
        elif dataset_type=="test":
            self.gens_to_ret =  self.params.test_gens
        
        print(f" Loading {dataset_type} Dataset")

        # can add more here, example granular_pop is not being used
        self.data = {'X':None, 'y':None, 'cps':None, 'superpop':None, 'granular_pop':None}

        if labels_path is None:
            print(f'Loading snps data')
            snps = load_path(osp.join(path_prefix, str(dataset_type),'mat_vcf_2d.npy'))
            self.data['X'] = torch.tensor(self.snps[:,0:self.params.chmlen])
            print(f"snps data shape : {self.data['X'].shape}")
        else:
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
   
            pop_sample_map = pd.read_csv(osp.join(labels_path, self.params.pop_sample_map), sep='\t')
            self.pop_arr = repeat_pop_arr(pop_sample_map)
            self.coordinates = load_path(osp.join(labels_path, self.params.coordinates), en_pickle=True)
            self.load_data()
    
    def __len__(self):
        return len(self.data['X']) 
    
    def mapping_func(self, arr, b, dim):
        """
        Inputs:
        arr: 3(d)D array
        b : dict with 3(d) dim array as values
        d: dimension of the output, could be 3 or more
        return:
        result: 3(d)D array
        """
        result = np.zeros((arr.shape[0], arr.shape[1], dim)).astype(float)
        
        for k in np.unique(arr):
            idx = np.nonzero(arr==k)
            for d in np.arange(dim):
                result[idx[0], idx[1], d]=b[k][d]
            
        result = torch.tensor(result).float()
        return result

    def pop_mapping(self, y_vcf, pop_arr, type='superpop'):
        
        result = np.zeros((y_vcf.shape[0], y_vcf.shape[1])).astype(float)
        if type=='superpop':
            col_num=3
            for k in np.unique(y_vcf):
                idx = np.nonzero(y_vcf==k)
                pop_arr_idx = np.nonzero(pop_arr[:,1]==k)[0]
                result[idx[0], idx[1]]=pop_arr[pop_arr_idx, col_num]
        result = torch.tensor(result).float()
        return result

    @timer
    def load_data(self):        
        # take the mode according to windows for labels
        # map to coordinates according to ref_idx
        print("Transforming the data")
        self.data['X'] = torch.tensor(self.snps[:,0:self.params.chmlen])
        y_tmp = torch.tensor(self.vcf_idx[:,0:self.params.chmlen])
        y_tmp = y_tmp.reshape(-1, self.params.n_win, self.params.win_size)
        self.y_vcf_idx = (torch.mode(y_tmp, dim=2)[0]).detach().cpu().numpy()
        
        self.data['y'] = self.mapping_func(self.y_vcf_idx, self.coordinates, self.params.dataset_dim)
        
        if self.params.cp_detect:
            # if the only gen is gen 0 then there will be no changepoints with founders
            assert max(self.gens_to_ret)>0, "No changepoints will be found. Feeding only founders."
            cps = self.data['y'][:,:-1,:]-self.data['y'][:,1:,:]
            #insert 0 at the end
            cps = torch.cat((cps, torch.zeros_like(self.data['y'][:,0,:]).unsqueeze(1)), dim=1)
            
            assert cps.sum()!=0, "No changepoints found. Check the input file"
            cp_idx = torch.nonzero(cps)[0]
            for i in cp_idx:
                cps[i-self.params.cp_tol:i+self.params.cp_tol] = cps[i]
            #revert to make a mask
            # self.data['cp_mask'] = torch.where(cps==0.0, torch.tensor([1.0]), torch.tensor([0.0]))
            self.data['cps'] = cps
        else:
            self.data['cps'] = torch.ones_like(self.data['y'])

        if self.params.superpop_mask:
            self.data['superpop'] = self.pop_mapping(self.y_vcf_idx, self.pop_arr, type='superpop')
        else:
            self.data['superpop'] = torch.ones_like(self.data['y'])
    
    def __getitem__(self, idx):
        ls =[]
        for k, v in self.data.items():
            if v is not None:
                ls.append(self.data[k][idx])
        return ls

