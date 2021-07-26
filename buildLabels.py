import numpy as np
import pandas as pd
import logging
import os.path as osp
import allel
from src.utils.labelUtil import filter_reference_file, get_sample_map, getLabelsPath,\
    repeat_pop_arr, split_sample_maps, createCoordinates, getAdmixedCombineChm, writeCompact
from src.utils.dataUtil import load_path, save_file, vcf2npy
from src.createLabels.unsupervisedMethods import pcaSpace, spectralEmbeddingSpace, umapSpace, \
    tsneSpace, residualPca, thinningPcaSpace
from src.createLabels.settings_labels import parse_args
from enum import Enum

def main(config):
    # print the configurations in the log directory
    for k, v in config.items():
        print(f"config for {k} : {v}")
        
    # set seed 
    seed = config['data.seed']
    np.random.seed(seed)
    print(f"seed used in this run : {seed}")

    data_out_path=getLabelsPath(config['data.geno_type'],config['data.expt_id'],config['data.method'])
    
    # Note1: ref_idx will contain the idx for haplotypes that are at this intersection
    # of sample map and the vcf file
    print("Extracting snps that are at the intersection of vcf samples and sample map")
    vcf_snp = allel.read_vcf(config['data.vcf_dir'][0])
    ref_sample_map = pd.read_csv(config['data.reference_map'], sep="\t")
    
    # select the intersection of snps between ref map and vcf
    sample_ref_vcf = pd.DataFrame(vcf_snp['samples'])
    sample_ref_vcf.columns=['Sample']
    sample_ref_vcf['ref_idx'] = sample_ref_vcf.index
    #merge sample_ref with ref_sample_map
    master_ref = ref_sample_map.merge(sample_ref_vcf, how="inner", on="Sample")

    if config['data.geno_type']=='humans':
        master_ref = filter_reference_file(master_ref)
    
    pop_sample_map, granular_pop_dict, superpop_dict = get_sample_map(master_ref)
    rev_pop_dict={v:k for k,v in granular_pop_dict.items()}
    rev_superpop_dict = {v:k for k,v in superpop_dict.items()}
    # save the above three
    save_file(osp.join(data_out_path, 'pop_sample_map.tsv'), pop_sample_map, en_df=True)
    save_file(osp.join(data_out_path, 'granular_pop.pkl'), granular_pop_dict, en_pickle=True)
    save_file(osp.join(data_out_path, 'superpop.pkl'), superpop_dict, en_pickle=True)
    
    #for labels
    # pop_arr_xx is defined with column 0: 'sample_id'
    # col 1: vcf_ref_idx, col2: granular_pop_num
    # col3: superpop_num

    pop_arr = repeat_pop_arr(pop_sample_map)
    if config['data.create_labels']:
        print(f"Creating labels using {config['data.method']}")
        
        if config['data.method'] == "geo":
            createCoordinates(pop_arr, master_ref, data_out_path)
        else:
            print("Loading vcf file for unsupervised methods")
            vcf_data = load_path(config['data.all_chm_snps']) if str(config['data.all_chm_snps'])!= 'None' else vcf2npy(config['data.vcf_dir'][0])
            # filter the vcf_data with the same snps
            vcf_data=vcf_data[pop_arr[:,1].astype(int),:]
            
            print(f"vcf shape for unsupervised method:{vcf_data.shape}")
            print("Computing labels")

            # define custom and default pop order, pop_num to use everywhere
            if config['data.pop_order'] is None:
                config['data.pop_order'] = list(superpop_dict.keys())

            assert config['data.method'].lower() in ["pca", "residual_pca", "tsne", "umap", \
                "spectral_embedding", "plink_pca", "geo"], "Invalid method selected"
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

        #save the sample_maps
        save_file(osp.join(data_out_path, 'train_sample_map.tsv'), train_sample_map, en_df=True)
        save_file(osp.join(data_out_path, 'valid_sample_map.tsv'), valid_sample_map, en_df=True)
        save_file(osp.join(data_out_path, 'test_sample_map.tsv'), test_sample_map, en_df=True)
        
        # save the train, valid and test sample maps
        # create admixed samples. Test data may not always be present.
        dataset_type = ['train', 'valid', 'test'] if len(test_sample_map)>2 else ['train', 'valid']
        sample_map_lst = [train_sample_map, valid_sample_map, test_sample_map] if len(test_sample_map)>2 else [train_sample_map, valid_sample_map]
        admixed_num_per_gen = config['data.samples_per_type']
        print("Forming the dataset with simulation")
        if len(config['singleAncestryAdmixture'])>1 :
            for p, valPop in config['singleAncestryAdmixture']:
                for i, val in enumerate(dataset_type):
                    #subset the sample_map for that pop
                    subsetSampleMap = sample_map_lst[i]
                    subsetSampleMap = subsetSampleMap.loc[subsetSampleMap.superpop==rev_superpop_dict[valPop],:]
                    start_chm=config['data.start_chm']
                    end_chm=config['data.end_chm']
                    print(f"Simulating for pop {valPop} for dataset {val} for start chm {start_chm} end chm {end_chm}")
                    save_path = osp.join(data_out_path, ''.join([str(val),'_', str(valPop)]))
                    genetic_map_path = str(config['data.genetic_map'])
                    getAdmixedCombineChm(start_chm, end_chm, genetic_map_path=genetic_map_path, vcf_founders=config['data.vcf_dir'], \
                    sample_map=subsetSampleMap, save_path=save_path, num_samples=admixed_num_per_gen, \
                    gens_to_ret=config['data.gens_to_ret'], pop_arr=pop_arr, sample_weight=config['sim.balanced'])
            # combine all the valpops into one folder, so it can be read downstream as dataset_type/gen
            writeCompact(config['data.gens_to_ret'], dataset_type, data_out_path)

        else:
            for i, val in enumerate(dataset_type):
                start_chm=config['data.start_chm']
                end_chm=config['data.end_chm']
                print(f"Simulating for dataset {val} for start chm {start_chm} end chm {end_chm}")
                save_path = osp.join(data_out_path, str(val))
                genetic_map_path = str(config['data.genetic_map'])
                getAdmixedCombineChm(start_chm, end_chm, genetic_map_path=genetic_map_path, vcf_founders=config['data.vcf_dir'], \
                sample_map=sample_map_lst[i], save_path=save_path, num_samples=admixed_num_per_gen, \
                gens_to_ret=config['data.gens_to_ret'], pop_arr=pop_arr, sample_weight=config['sim.balanced'])
        
if __name__=="__main__":
    config = parse_args()
    main(config)