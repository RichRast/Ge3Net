import numpy as np
import pandas as pd
import sys 
import os
sys.path.insert(1,os.environ.get('USER_PATH'))
from pyadmix.utils import get_chm_info, build_founders, create_non_rec_dataset, write_output 

def filter_reference_file(ref_sample_map, verbose=True):
    """
    read the reference file and filter by default criteria of single_ancestry =1
    for humans. Add other criteria to filter here
    """
    print(f" Total samples before filtering : {len(ref_sample_map)}")
    ref_sample_map = ref_sample_map[ref_sample_map['Single_Ancestry']==1].reset_index(drop=True)
    
    if verbose:
        print(f"Total {len(ref_sample_map)} number of samples selected")
    return ref_sample_map


def get_sample_map(sample_map, pop_order):
    """
    Reads the sample map and returns a tsv file with 
    Sample: unique sample ID
    ref_idx: index from reference file 
    superpop: superpop out of the 7 continents, range 0-6
    granularpop: granular ancestries, range 0-135
    """
    granular_pop_arr = sample_map['Population'].unique()
    granular_pop_dict = {k:v for k,v in zip(granular_pop_arr, range(len(granular_pop_arr)))}
    superpop_arr = sample_map['Superpopulation code'].unique()
    superpop_dict = {k:v for k,v in zip(superpop_arr, range(len(superpop_arr)))}
    pop_sample_map = sample_map.loc[:,['Sample','ref_idx']]
    pop_sample_map['granular_pop'] = list(map(lambda x:granular_pop_dict[x], sample_map['Population'].values))
    pop_sample_map['superpop'] = list(map(lambda x:superpop_dict[x], sample_map['Superpopulation code'].values))
    
    return pop_sample_map, granular_pop_dict, superpop_dict

def split_sample_maps(sample_map, split_perc, random_seed=10):
    """
    This function splits the sample_map with single ancestry =1 into train, val and test splits
    whose split ratio is given by split_perc

    sample_map : sample map with column names: 'Sample', 'granular_pop', 'ref_idx' and 'superpop'
    'Sample' is the sample id, 'granular_pop; is the granular population number 
    split_perc: [train_perc, valid_perc, test_perc]
    return : tsv for train, valid and test sample maps
    """
    np.random.seed(random_seed)
    # find the numbers of samples to split into train, val and test
    df_pop_count = sample_map[['Sample', 'granular_pop']].groupby(['granular_pop']).count()
    train_perc, valid_perc, test_perc = split_perc[0], split_perc[1], split_perc[2]
    total_count = df_pop_count['Sample'].values
    df_pop_count['Train']=np.rint(train_perc*total_count).astype(int)
    df_pop_count['Valid']=np.rint((valid_perc/(1-train_perc))*(total_count - df_pop_count['Train'])).astype(int)
    df_pop_count['Test']=total_count - (df_pop_count['Train'] + df_pop_count['Valid']).astype(int)
    df_pop_count.reset_index(inplace=True)
    
    l = [None]*3
    for i, pop_val in enumerate(df_pop_count['granular_pop'].values):  
        train_count = df_pop_count[df_pop_count['granular_pop']==pop_val]['Train'].values.item()
        val_count = df_pop_count[df_pop_count['granular_pop']==pop_val]['Valid'].values.item()
        sample_ids = sample_map[sample_map['granular_pop']==pop_val].values
        np.random.shuffle(sample_ids)
        curr = np.split(sample_ids, [train_count, train_count + val_count])
        for j in range(len(curr)): l[j] = np.concatenate((l[j], curr[j])) if i>0 else curr[j]
        
    train_sample_map = pd.DataFrame(l[0], columns=sample_map.columns)
    valid_sample_map = pd.DataFrame(l[1], columns=sample_map.columns)
    test_sample_map = pd.DataFrame(l[2], columns=sample_map.columns)
    
    return train_sample_map, valid_sample_map, test_sample_map

def get_admixed_samples(genetic_map_path, vcf_founders, sample_map, save_path, num_samples, gens_to_ret, random_seed=10):
    """
    Use XGMix simulation to create admixed dataset
    """
    genetic_map = get_chm_info(genetic_map_path, vcf_founders)
    founders = build_founders(vcf_founders, genetic_map, sample_map)
    admixed_samples, select_idx = create_non_rec_dataset(founders, \
                    num_samples, gens_to_ret, genetic_map["breakpoint_probability"], random_seed)
    write_output(save_path, admixed_samples)
    return admixed_samples, select_idx

def repeat_pop_arr(sample_map):
    """
    This function maps from ref idx of sample map
    to ref idx of vcf file by repeating for 2*i and 2*i+1
    """
    pop_arr = sample_map.values[:, np.newaxis, :]
    pop_arr = np.repeat(pop_arr, 2, axis=1)
    pop_arr = pop_arr.reshape(2*len(sample_map),-1)
    pop_arr[:,1] = [i for x in sample_map.values[:,1] for i in (2*x, 2*x+1)]
    return pop_arr