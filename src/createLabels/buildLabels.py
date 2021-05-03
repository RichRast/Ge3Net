import numpy as np
import pandas as pd
import os
import os.path as osp
import allel
import sys
sys.path.insert(1, os.environ.get('USER_PATH'))
from src.utils.labelUtil import filter_reference_file, get_sample_map, \
    repeat_pop_arr, split_sample_maps, get_admixed_samples
from src.utils.dataUtil import load_path, save_file, vcf2npy
from unsupervisedMethods import pcaSpace, spectralEmbeddingSpace, umapSpace, \
    tsneSpace, residualPca, thinningPcaSpace
from settings import parse_args

def main(config):
    # print the configurations in the log directory
    for k, v in config.items():
        print(f"config for {k} : {v}")
        
    # set seed 
    seed = config['data.seed']
    np.random.seed(seed)
    print(f"seed used in this run : {seed}")

    # check values for geno_type
    assert config['data.geno_type'] in ['humans', 'dogs'], " invalid value of geno type"

    data_out_path = osp.join(str(config['data.data_out']), \
                            str(config['data.geno_type']), \
                            'labels' , \
                ''.join(["data_id_", str(config['data.expt_id']), "_", str(config['data.method'])]))
    print(f"data_out_path : {str(data_out_path)}")
    if not osp.exists(data_out_path):
        print(f"dataset out dir doesn't exist, making {str(data_out_path)}")
        os.makedirs(data_out_path , exist_ok=True)

    # Note1: Throughout vcf_idx refers to 2i and 2i+1 
    # Note1: and ref_idx refers to the reference idx in reference sample map
    print("Reading reference map")

    vcf_snp = allel.read_vcf(config['data.vcf_dir'])
    ref_sample_map = pd.read_csv(config['data.reference_map'], sep="\t")
    
    #select the intersection of snps between ref map and vcf
    sample_ref_vcf = pd.DataFrame(vcf_snp['samples'])
    sample_ref_vcf.columns=['Sample']
    sample_ref_vcf['ref_idx'] = sample_ref_vcf.index
    #merge sample_ref with ref_sample_map
    ref_sample_map = ref_sample_map.merge(sample_ref_vcf, how="inner", on="Sample")

    if config['data.geno_type']=='humans':
        master_ref = filter_reference_file(ref_sample_map)
    else:
        master_ref = ref_sample_map
    pop_sample_map, granular_pop_dict, superpop_dict = get_sample_map(master_ref, config['data.pop_order'])
    rev_pop_dict={v:k for k,v in granular_pop_dict.items()}
    # save the above three
    save_file(osp.join(data_out_path, 'pop_sample_map.tsv'), pop_sample_map, en_df=True)
    save_file(osp.join(data_out_path, 'granular_pop.pkl'), granular_pop_dict, en_pickle=True)
    save_file(osp.join(data_out_path, 'superpop.pkl'), superpop_dict, en_pickle=True)
    
    if config['data.create_labels']:
        print("Loading vcf file for unsupervised methods")
        if str(config['data.all_chm_snps'])!= 'None':
            vcf_data = load_path(config['data.all_chm_snps'])
        else:
            vcf_data = vcf2npy(config['data.vcf_dir'])
        print(f"vcf shape for unsupervised method:{vcf_data.shape}")

        config['data.n_way'] = len(superpop_dict.items())
        pop_arr = repeat_pop_arr(pop_sample_map)
        
        print("Computing labels")

        # define custom and default pop order, pop_num to use everywhere
        if config['data.pop_order'] is None:
            config['data.pop_order'] = list(superpop_dict.keys())

        print(f"Creating labels using {config['data.method']}")
        assert config['data.method'].lower() in ["pca", "residual_pca", "tsne", "umap", \
            "spectral_embedding", "plink_pca"], "Invalid method selected"
        if config['data.method'].lower()=="plink_pca":
            unsupMethod=thinningPcaSpace(data_out_path, master_ref)
            sh_args=['whole_genome', config['data.sample_map'], config['data.geno_type']]
            plink_path=osp.join(str(config['data.data_out']), config['data.geno_type'],\
                'plink', ''.join(['sm_', str(config['data.sample_map'])]))
            unsupMethod(sh_args, plink_path)
        else:    
            if config['data.method'].lower()=="residual_pca":
                unsupMethod=residualPca(config['data.n_comp_overall'], data_out_path, \
                config['data.n_comp_overall'], config['data.n_comp_subclass'],\
                pop_arr, config['data.pop_order'],config['data.seed'])
            elif config['data.method'].lower()=="pca":
                unsupMethod=pcaSpace(config['data.n_comp_overall'], data_out_path, \
                config['data.pop_order'], config['data.seed'])
            elif config['data.method'].lower()=="umap":
                unsupMethod=umapSpace(config['data.n_comp_overall'], data_out_path, \
                config['data.pop_order'], config['data.seed'])
            elif config['data.method'].lower()=="tsne":
                unsupMethod=tsneSpace(config['data.n_comp_overall'], data_out_path, \
                config['data.pop_order'], config['data.seed'])
            elif config['data.method'].lower()=="spectral_embedding":
                unsupMethod=spectralEmbeddingSpace(config['data.n_comp_overall'], data_out_path, \
                config['data.pop_order'], config['data.seed'])
            unsupMethod(vcf_data, rev_pop_dict, pop_arr)
                    
    if config['data.simulate']:
        print("Split into train, valid and test data")
        #split into train, val and test dataset
        split_perc = config['data.split_perc']

        # to get a different split, set the config['data.seed'] argument before running the script
        train_sample_map, valid_sample_map, test_sample_map = split_sample_maps(pop_sample_map, split_perc, seed)

        #for train labels
        # pop_arr_xx is defined with column 0: 'sample_id'
        # col 1: vcf_ref_idx, col2: granular_pop_num
        # col3: superpop_num
        pop_arr_train = repeat_pop_arr(train_sample_map)
        train_vcf_idx = list(pop_arr_train[:,1])
        
        #for valid labels
        pop_arr_valid = repeat_pop_arr(valid_sample_map)
        valid_vcf_idx = list(pop_arr_valid[:,1])
        
        #for test labels
        pop_arr_test = repeat_pop_arr(test_sample_map)
        test_vcf_idx = list(pop_arr_test[:,1])

        #save the sample_maps
        save_file(osp.join(data_out_path, 'train_sample_map.tsv'), train_sample_map, en_df=True)
        save_file(osp.join(data_out_path, 'valid_sample_map.tsv'), valid_sample_map, en_df=True)
        save_file(osp.join(data_out_path, 'test_sample_map.tsv'), test_sample_map, en_df=True)

        # save the train, valid and test sample maps
        # create admixed samples
        dataset_type = ['train', 'valid', 'test']
        sample_map_lst = [train_sample_map, valid_sample_map, test_sample_map]
        idx_lst = [train_vcf_idx, valid_vcf_idx, test_vcf_idx]
        admixed_num_per_gen = config['data.samples_per_type']

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
    config = parse_args()
    main(config)