import argparse
import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
import allel

sys.path.insert(1, '/home/users/richras/GeNet_Repo')
from helper_funcs import load_path, save_file, vcf2npy
from unsupervised_methods import PCA_space
from visualization import plot_embeddings
from settings import parse_args

sys.path.insert(1, '/home/users/richras/XGMix/XGMix')
from pyadmix.utils import get_chm_info, build_founders, create_non_rec_dataset, write_output 


def filter_reference_file(ref_file_path, verbose=True):
    """
    read the reference file and filter by default criteria of single_ancestry =1
    """
    ref_sample_map = pd.read_csv(ref_file_path, sep="\t")
    ref_sample_map['ref_idx'] = ref_sample_map.index
    ref_sample_map = ref_sample_map[ref_sample_map['Single_Ancestry']==1].reset_index(drop=True)
    
    if verbose:
        print(f"Total {len(ref_sample_map)} number of samples selected")
    return ref_sample_map


def get_sample_map(sample_map):
    """
    Reads the sample map and returns an tsv file with 
    sample ID: unique sample ID
    ref_idx: index from reference file 
    superpop: superpop out of the 7 continents, range 0-6
    granularpop: granular ancestries, range 0-135
    """
    granular_pop_arr = sample_map['Population'].unique()
    granular_pop_dict = {k:v for k,v in zip(granular_pop_arr, range(len(granular_pop_arr)))}
    
    pop = ['EAS', 'SAS', 'WAS', 'OCE', 'AFR', 'AMR', 'EUR']
    superpop_dict = {k:v for k,v in zip(pop, range(len(pop)))}

    pop_sample_map = sample_map.loc[:,['Sample','ref_idx']]
    pop_sample_map['granular_pop'] = list(map(lambda x:granular_pop_dict[x], sample_map['Population'].values))
    pop_sample_map['superpop'] = list(map(lambda x:superpop_dict[x], sample_map['Superpopulation code'].values))
    
    return pop_sample_map, granular_pop_dict, superpop_dict

def split_sample_maps(sample_map, split_perc, random_seed=10):
    """
    This function splits the sample_map with single ancestry =1 into train, val and test splits
    whose split ratio is given by split_perc

    sample_map : sample map
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
    
    for i, pop_val in enumerate(df_pop_count['granular_pop'].values):
        
        train_count = df_pop_count[df_pop_count['granular_pop']==pop_val]['Train'].values.item()
        val_count = df_pop_count[df_pop_count['granular_pop']==pop_val]['Valid'].values.item()
        sample_ids = sample_map[sample_map['granular_pop']==pop_val].values
        np.random.shuffle(sample_ids)
        
        curr_l0, curr_l1, curr_l2 = np.split(sample_ids, [train_count, train_count + val_count])
        
        if i>0:
            l0 = np.concatenate((l0, curr_l0))
            l1 = np.concatenate((l1, curr_l1))
            l2 = np.concatenate((l2, curr_l2))
        else:
            l0 = curr_l0
            l1 = curr_l1
            l2 = curr_l2
        
    train_sample_map = pd.DataFrame(l0, columns=sample_map.columns)
    valid_sample_map = pd.DataFrame(l1, columns=sample_map.columns)
    test_sample_map = pd.DataFrame(l2, columns=sample_map.columns)
    
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

def main(config):

    # set seed 
    seed = config['train.seed']
    np.random.seed(seed)

    dataset_path = osp.join(str(config['data.working_dir']), config['data.experiment_name'], str(config['data.experiment_id']))
    if not osp.exists(dataset_path):
        os.makedirs(dataset_path , exist_ok=True)

    # Note1: Throughout vcf_idx and filter_idx refer to 2i and 2i+1 
    # Note1: and ref_idx refers to the reference idx in reference sample map

    print("Reading reference map")
    master_ref = filter_reference_file(str(config['data.reference_map']))
    filter_idx = list(chain.from_iterable((2*i ,(2*i)+1) for i in master_ref['ref_idx'].values))
    pop_sample_map, granular_pop_dict, superpop_dict = get_sample_map(master_ref)
    
    # save the above three
    save_file(osp.join(dataset_path, 'pop_sample_map.tsv'), pop_sample_map, en_df=True)
    save_file(osp.join(dataset_path, 'granular_pop.pkl'), granular_pop_dict, en_pickle=True)
    save_file(osp.join(dataset_path, 'superpop.pkl'), superpop_dict, en_pickle=True)
    
    if config['data.form_labels']:
        print("computing labels from PCA/MDS")
        if config['data.all_chm_snps'] is not None:
            vcf_snp = load_path(config['data.all_chm_snps'])
        else:
            vcf_snp = vcf2npy(str(config['data.vcf']))
        vcf_filtered = vcf_snp[filter_idx]
        
        PCA_labels = PCA_space(vcf_filtered, n_comp=5)
        PCA_lbls_dict = {k:v for k,v in zip(filter_idx, PCA_labels[:,0:4])}
        # save PCA labels corresponding to the reference index
        save_file(osp.join(dataset_path, 'PCA_labels.pkl'), PCA_lbls_dict, en_pickle=True)
        
        #plot PCA
        #repeat for maternal/paternal for founders
        pop_arr = pop_sample_map.values[:, np.newaxis, :]
        pop_arr = np.repeat(pop_arr, 2, axis=1)
        pop_arr = pop_arr.reshape(-1,4)
        #update the ref_idx to vcf_idx
        for j in range(len(pop_arr)):
            if j%2==0:
                pop_arr[j,1] = 2*pop_arr[j,1]
            else:
                pop_arr[j,1] = 2*pop_arr[j,1]+1

        ax = plot_embeddings(PCA_labels[:, 0:3], pop_arr[:,3], config['data.n_way'])
        #print randomly 30 granular pop on the PCA plot
        random_idx = np.random.choice(filter_idx, 30)
        rev_pop_order={v:k for k,v in granular_pop_dict.items()}
        for i in random_idx:
            idx_pop_arr=np.where(pop_arr[:,1]==i)[0][0]
            ax.text(PCA_lbls_dict[i][0],PCA_lbls_dict[i][1],PCA_lbls_dict[i][2], \
                    s = rev_pop_order[pop_arr[idx_pop_arr,2]],\
                fontweight='bold', fontsize = 12)
        plt.show()
    
    if config['data.simulate']:
        print("Forming the dataset with simulation")
        #split into train, val and test dataset
        split_perc = config['data.split_perc']
        train_sample_map, valid_sample_map, test_sample_map = split_sample_maps(pop_sample_map, split_perc)
        train_idx = train_sample_map['ref_idx'].values.astype(int)
        valid_idx = valid_sample_map['ref_idx'].values.astype(int)
        test_idx = test_sample_map['ref_idx'].values.astype(int)

        # save the train, valid and test sample maps
        # create admixed samples
        dataset_type = ['train', 'valid', 'test']
        sample_map_lst = [train_sample_map, valid_sample_map, test_sample_map]
        idx_lst = [train_idx, valid_idx, test_idx]
        admixed_num_per_gen = config['data.samples_per_type']

        selected_idx={}
        for i, val in enumerate(dataset_type):
            save_path = osp.join(dataset_path, str(val))
            genetic_map_path = str(config['data.genetic_map'])
            if (config['data.vcf'][-4:]=='.vcf') or (config['data.vcf'][-3:]=='.gz'):
                vcf_master = allel.read_vcf(str(config['data.vcf']))
            else:
                vcf_master = load_path(config['data.vcf'], en_pickle=True)
            _, selected_idx[val] = get_admixed_samples(genetic_map_path, vcf_master, 
            sample_map_lst[i], save_path, admixed_num_per_gen[i], config['data.gens_to_ret'])

if __name__=="__main__":
    config,_ = parse_args()
    main(config)