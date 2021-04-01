import argparse
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
import allel
import wandb

from helper_funcs import load_path, save_file, vcf2npy
from unsupervised_methods import PCA_space_residual
from visualization import plot_embeddings, plot_subclass
from settings import parse_args

from pyadmix.utils import get_chm_info, build_founders, create_non_rec_dataset, write_output 

# POP_ORDER = ['EAS', 'SAS', 'WAS', 'OCE', 'AFR', 'AMR', 'EUR']

def create_ref_sample_map(metadata_filename, sa_sample_filename):
    """
    create reference sample map for dogs, with columns = 'Sample',
    'Population' for granular pop and 'Superpopulation code' for superpop.
    These samples have been pre-filetered with the criteria of Single Ancestry = 1
    """
    metadata = pd.read_excel(open(metadata_filename, 'rb'), engine='openpyxl')
    sa_sample = pd.read_csv(sa_sample_filename, delimiter = "\t", header=None)
    sa_sample.columns=['Sample', 'Population']
    ref_map_filtered = metadata[np.in1d(metadata['Name_ID'].values, sa_sample['Sample'])]
    ref_sample_map = ref_map_filtered[['Name_ID', 'Breed/CommonName']]

    ref_sample_map = ref_sample_map.merge(sa_sample[['Sample','Population']], how='left', \
        left_on='Name_ID', right_on='Sample')
    ref_sample_map.drop(columns=['Sample'], inplace=True)
    ref_sample_map.rename(columns={'Name_ID': 'Sample', 'Population':'Superpopulation code', \
                               'Breed/CommonName': 'Population'}, inplace=True)

    return ref_sample_map

def filter_reference_file(ref_sample_map, criteria, geno_type, verbose=True):
    """
    read the reference file and filter by default criteria of single_ancestry =1
    for humans. Add other criteria to filter here
    """
    ref_sample_map['ref_idx'] = ref_sample_map.index
    print(f" Total samples before filtering : {len(ref_sample_map)}")
    if geno_type == 'humans':
        if criteria == 'Single_Ancestry':
            ref_sample_map = ref_sample_map[ref_sample_map['Single_Ancestry']==1].reset_index(drop=True)
    elif geno_type == 'dogs':
        if criteria == 'rm_anc':
            rm_ancestry = ['Wolf', 'Coyote']
            ref_sample_map = ref_sample_map[~ref_sample_map['Superpopulation code'].isin(rm_ancestry)]
    
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

def main(config):
    
    # print the configurations in the log directory
    # for k, v in config.items():
    #     print(f"config for {k} : {v}")
        
    # set seed 
    seed = config['data.seed']
    np.random.seed(seed)
    print(f"seed used in this run : {seed}")

    # check values for geno_type
    assert config['data.geno_type'] in ['humans', 'dogs'], " invalid value of geno type"

    data_out_path = osp.join(str(config['data.data_out']), config['data.geno_type'], config['data.experiment_name'], str(config['data.experiment_id']))
    print(f"data_out_path : {str(data_out_path)}")
    if not osp.exists(data_out_path):
        print(f"dataset out dir doesn't exist, making {str(data_out_path)}")
        os.makedirs(data_out_path , exist_ok=True)

    try:
        wandb.init(project="Build_labels")
    except Exception as e:
        print(e)
        print("wandb could not be initialized")
        wandb = None
    else:
        print("logging to wandb")

    # Note1: Throughout vcf_idx and filter_idx refer to 2i and 2i+1 
    # Note1: and ref_idx refers to the reference idx in reference sample map
    print("Reading reference map")

    # check values for ref_filter_criteria
    assert config['data.ref_filter_criteria'] in ['Single_Ancestry', 'rm_anc'], " invalid filter criteria"

    if config['data.geno_type']=='humans':
        ref_sample_map = pd.read_csv(config['data.reference_map'], sep="\t")
    elif config['data.geno_type']=='dogs':
        ref_sample_map = create_ref_sample_map(config['data.metadata'], config['data.reference_map'])

    master_ref = filter_reference_file(ref_sample_map, config['data.ref_filter_criteria'], \
        config['data.geno_type'])
    pop_sample_map, granular_pop_dict, superpop_dict = get_sample_map(master_ref, config['data.pop_order'])
    
    # save the above three
    save_file(osp.join(data_out_path, 'pop_sample_map.tsv'), pop_sample_map, en_df=True)
    save_file(osp.join(data_out_path, 'granular_pop.pkl'), granular_pop_dict, en_pickle=True)
    save_file(osp.join(data_out_path, 'superpop.pkl'), superpop_dict, en_pickle=True)
    
    if config['data.create_labels']:
        print("Split into train, valid and test data")
        #split into train, val and test dataset
        split_perc = config['data.split_perc']

        # to get a different split, set the config['data.seed'] argument before running the script
        train_sample_map, valid_sample_map, test_sample_map = split_sample_maps(pop_sample_map, split_perc, seed)

        print("Loading vcf file for PCA/MDS")
        if str(config['data.all_chm_snps'])!= 'None':
            vcf_data = load_path(config['data.all_chm_snps'])
        else:
            vcf_data = vcf2npy(config['data.vcf_dir'])

        vcf_snp = allel.read_vcf(config['data.vcf_dir'])
        #for train labels
        # pop_arr_xx is defined with column 0: 'sample_id'
        # col 1: vcf_ref_idx, col2: granular_pop_num
        # col3: superpop_num
        
        # match the sample id of pop_arr_train and vcf file
        # and get the train_vcf_idx as the indices in the vcf file that match the sample id of train sample map
        
        train_sample_map = train_sample_map[np.in1d(train_sample_map['Sample'], vcf_snp['samples'])]
        pop_arr_train = repeat_pop_arr(train_sample_map)
        train_vcf_idx = list(pop_arr_train[:,1])
        
        #for valid labels
        valid_sample_map = valid_sample_map[np.in1d(valid_sample_map['Sample'], vcf_snp['samples'])]
        pop_arr_valid = repeat_pop_arr(valid_sample_map)
        valid_vcf_idx = list(pop_arr_valid[:,1])
        
        #for test labels
        test_sample_map = test_sample_map[np.in1d(test_sample_map['Sample'], vcf_snp['samples'])]
        pop_arr_test = repeat_pop_arr(test_sample_map)
        test_vcf_idx = list(pop_arr_test[:,1])

        # save the train, valid and test sample maps
        # create admixed samples
        dataset_type = ['train', 'valid', 'test']
        sample_map_lst = [train_sample_map, valid_sample_map, test_sample_map]
        idx_lst = [train_vcf_idx, valid_vcf_idx, test_vcf_idx]
        admixed_num_per_gen = config['data.samples_per_type']

        config['data.n_way'] = len(superpop_dict.items())

        print("Computing labels from PCA/MDS")
        # pca_train is the pca object to be used for transform for new data
        n_comp_overall=3
        n_comp_subclass=2
        n_comp=44

        # define custom and default pop order, pop_num to use everywhere
        if config['data.pop_order'] is None:
            config['data.pop_order'] = list(superpop_dict.keys())

        print("Computing pca")
        
        PCA_labels_train, PCA_labels_valid, PCA_labels_test, pca_train = \
            PCA_space_residual(vcf_data, idx_lst, config['data.pop_order'], n_comp=config['data.n_comp'], n_comp_overall=config['data.n_comp_overall'], \
            extended_pca = config['data.extended_pca'], pop_arr=pop_arr_train[:,3], n_way = config['data.n_way'],\
            n_comp_subclass = config['data.n_comp_subclass'])
        
        PCA_lbls_train_dict = {k:v for k,v in zip(train_vcf_idx, PCA_labels_train)}
        PCA_lbls_valid_dict = {k:v for k,v in zip(valid_vcf_idx, PCA_labels_valid)}
        PCA_lbls_test_dict = {k:v for k,v in zip(test_vcf_idx, PCA_labels_test)}

        PCA_lbls_dict = {**PCA_lbls_train_dict, **PCA_lbls_valid_dict, **PCA_lbls_test_dict}
        
        # save PCA labels corresponding to the reference index
        save_file(osp.join(data_out_path, 'PCA_labels_train.pkl'), PCA_lbls_train_dict, en_pickle=True)
        save_file(osp.join(data_out_path, 'PCA_labels_valid.pkl'), PCA_lbls_valid_dict, en_pickle=True)
        save_file(osp.join(data_out_path, 'PCA_labels_test.pkl'), PCA_lbls_test_dict, en_pickle=True)
        save_file(osp.join(data_out_path, 'PCA_labels.pkl'), PCA_lbls_dict, en_pickle=True)
        # save_file(osp.join(data_out_path, 'PCA_train.pkl'), pca_train, en_pickle=True)

        #save the sample_maps
        save_file(osp.join(data_out_path, 'train_sample_map.tsv'), train_sample_map, en_df=True)
        save_file(osp.join(data_out_path, 'valid_sample_map.tsv'), valid_sample_map, en_df=True)
        save_file(osp.join(data_out_path, 'test_sample_map.tsv'), test_sample_map, en_df=True)
            
        # print randomly 30 granular pop on the PCA plot of train 
        # random_idx refers to vcf_ref_idx
        random_idx = np.random.choice(train_vcf_idx, 30)
        # dict with key as granular_pop_num, and value as string of granular_pop
        rev_pop_order={v:k for k,v in granular_pop_dict.items()}
        ax, fig1 = plot_embeddings(n_comp_overall, pop_arr_train, config['data.n_way'], \
            random_idx, rev_pop_order, config['data.pop_order'], PCA_lbls_train_dict )
        plt.title('train_overall_pca')
        plt.show()
        fig1.savefig(osp.join(data_out_path, 'train overall_pca.png'))

        random_idx = np.random.choice(valid_vcf_idx, 30)
        ax, fig2 = plot_embeddings(n_comp_overall, pop_arr_valid, config['data.n_way'], \
            random_idx, rev_pop_order, config['data.pop_order'], PCA_lbls_valid_dict) 
        plt.title('valid_overall_pca')
        plt.show()
        fig2.savefig(osp.join(data_out_path, 'valid overall_pca.png'))

        random_idx = np.random.choice(test_vcf_idx, 30)
        ax, fig3 = plot_embeddings(n_comp_overall, pop_arr_test, config['data.n_way'], \
            random_idx, rev_pop_order, config['data.pop_order'], PCA_lbls_test_dict)
        plt.title('test_overall_pca')
        plt.show()
        # save .png for overall pca
        fig3.savefig(osp.join(data_out_path, 'test overall_pca.png'))

        if config['data.extended_pca']:
            # pop_order = list(superpop_dict.keys())
            # pop_order = ['EAS', 'SAS', 'WAS', 'OCE', 'AFR', 'AMR', 'EUR']
            # pop_num = np.arange(len(config['data.pop_order']))
            # pop_num = [4,6,2,1,0,3,5]

            print("Train subclasses")
            plot_subclass(config['data.pop_order'], pop_arr_train, PCA_lbls_train_dict, n_comp_overall, n_comp_subclass,\
                 rev_pop_order, wandb)
            print("Valid subclasses")
            plot_subclass(config['data.pop_order'], pop_arr_valid, PCA_lbls_valid_dict, n_comp_overall, n_comp_subclass, \
                rev_pop_order, wandb)
            print("Test subclasses")
            plot_subclass(config['data.pop_order'], pop_arr_test, PCA_lbls_test_dict, n_comp_overall, n_comp_subclass,\
                 rev_pop_order, wandb)
                    
        if wandb is not None:
            fig_image_train_overall = wandb.Image(fig1)
            fig_image_valid_overall = wandb.Image(fig2)
            fig_image_test_overall = wandb.Image(fig3)
            wandb.log({f"PCA train overall plot":fig_image_train_overall})
            wandb.log({f"PCA valid overall plot":fig_image_valid_overall})
            wandb.log({f"PCA test overall plot":fig_image_test_overall})
        plt.close('all')
    
    if config['data.simulate']:
        print("Forming the dataset with simulation")
        
        selected_idx={}
        for i, val in enumerate(dataset_type):
            save_path = osp.join(data_out_path, str(val))
            genetic_map_path = str(config['data.genetic_map'])
            if (config['data.vcf_dir'][-4:]=='.vcf') or (config['data.vcf_dir'][-3:]=='.gz'):
                vcf_master = allel.read_vcf(str(config['data.vcf_dir']))
            else:
                vcf_master = load_path(config['data.vcf_dir'], en_pickle=True)
            _, selected_idx[val] = get_admixed_samples(genetic_map_path, vcf_master, 
            sample_map_lst[i], save_path, admixed_num_per_gen[i], config['data.gens_to_ret'])

if __name__=="__main__":
    config,_ = parse_args()
    main(config)