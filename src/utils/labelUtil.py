import numpy as np
import pandas as pd
import scipy
import logging
import os
import allel
import os.path as osp
from pyadmix.utils import get_chm_info, build_founders, create_non_rec_dataset, write_output 
from src.utils.dataUtil import save_file, getValueBySelection, load_path
from src.utils.decorators import timer
from src.utils.modelUtil import convert_nVector
from typing import List

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

def get_sample_map(sample_map):
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

# deprecate the below to be replaced by the more general getAdmixedCombineChm
def get_admixed_samples(genetic_map_path, vcf_founders, sample_map, save_path, \
    num_samples, gens_to_ret, random_seed=10):
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

def createCoordinates(pop_arr, ref_map, save_path):
    """
    This function forms the labels.pkl ({ref_idx:labels})
    and labelsBysample.tsv
    (Sample, ref_idx, labels)
    for human geography from the sample map and the master
    reference map
    """
    df_labels= pd.DataFrame(pop_arr, columns=['Sample', 'ref_idx', 'granular_pop', 'superpop'])
    df_labels=df_labels.merge(ref_map[['Sample', 'Latitude', 'Longitude']], how="inner", on="Sample").reset_index(drop=True)
    # save labelsBySample.tsv
    df_labelsBySample=df_labels[['Sample', 'ref_idx', 'Latitude', 'Longitude']]
    df_labelsBySample['labels']=list(df_labels[['Latitude', 'Longitude']].to_numpy())
    df_labelsBySample.drop(columns=['Latitude', 'Longitude'], inplace=True)
    df_labelsBySample.to_csv(osp.join(save_path, "labelsBySample.tsv"), sep="\t", index=None)
    # save labels.pkl as a dict with vcf_idx as key and coordinates as values
    lbls_dict={k:v for k,v in zip(df_labels['ref_idx'], df_labelsBySample['labels'])}
    save_file(osp.join(save_path, 'labels.pkl'), lbls_dict, en_pickle=True)

def getLabelsPath(geno_type, expt_id, method):
    data_out_path = osp.join(os.environ.get('OUT_PATH'), str(geno_type), \
    'labels', ''.join(["data_id_", str(expt_id), "_", str(method)]))
    logging.info(f"data_out_path : {str(data_out_path)}")
    if not osp.exists(data_out_path):
        logging.info(f"dataset out dir doesn't exist, making {str(data_out_path)}")
        os.makedirs(data_out_path , exist_ok=True)
    return data_out_path

def getSampleNames(pop_arr: np.ndarray, admixedY: np.ndarray)-> List :

    samples =[]
    for i in range(admixedY.shape[0]):
        admixedSample=admixedY[i,:].reshape(-1,1)
        admixedSampleNames=list(map(lambda x: getValueBySelection(pop_arr, 1, x, 0), admixedSample))
        samples.append(admixedSampleNames)
    return samples

def getAdmixedCombineChm(*args, **kwargs):
    """
    In order to run it for the usual admixture for a single chm, 
    pass start_chm=end_chm= the desired chm number
    """
    start_chm=args[0]
    end_chm=args[1]
    genetic_map_path=kwargs.get('genetic_map_path') # one genetic map path for all chm
    vcf_founders=kwargs.get('vcf_founders') # list of vcf_founders for all chms
    sample_map=kwargs.get('sample_map')
    num_samples=kwargs.get('num_samples')
    gens_to_ret=kwargs.get('gens_to_ret')
    random_seed=kwargs.get('random_seed')
    save_path=kwargs.get('save_path')

    prevAdmixedFlag=True if end_chm>start_chm else False
    print(f"prevAdmixedFlag:{prevAdmixedFlag}")

    for i, chm in enumerate(range(start_chm-1, end_chm)):
        save_path_chm=osp.join(save_path, ''.join(['chm', str(chm+1)])) if prevAdmixedFlag else save_path
        print(f"i:{i}, chm:{chm}")
        if (vcf_founders[i][-4:]=='.vcf') or (vcf_founders[i][-3:]=='.gz'):
            vcf_master = allel.read_vcf(str(vcf_founders[i]))
        else:
            vcf_master = load_path(vcf_founders[i], en_pickle=True)
        genetic_map = get_chm_info(genetic_map_path, vcf_master)
        founders, foundersIdx = build_founders(vcf_master, genetic_map, sample_map)
        if chm==start_chm-1:
            admixed_samples_start, select_idx = create_non_rec_dataset(founders, \
                            num_samples, gens_to_ret, genetic_map["breakpoint_probability"], random_seed)
            write_output(save_path_chm, admixed_samples_start)
        else:
            admixed_samples, select_idx = create_non_rec_dataset(founders, num_samples, gens_to_ret,\
            genetic_map["breakpoint_probability"], random_seed, prevAdmixedFlag=prevAdmixedFlag,\
                 prevAdmixed=admixed_samples_start, foundersIdx=foundersIdx)
            write_output(save_path_chm, admixed_samples)

@timer   
def nearestNeighbourMapping(labels_path:str, preds: np.ndarray, labelType="superpop", distType="L2")->np.ndarray:
    """
    compute distance of preds from labels and assign the bin to the closest label
    for a single example or a batch of examples
    Input:
        preds: shape (n_samples*n_win)x(n_dim)
        pop_arr: [sample, vcf_idx, granular_pop, superpop]
        labels_path: dir name for labels.pkl
    Returns:
        mappedSpArr: shape (n_samples*n_win)
    """
    labels_all=load_path(osp.join(labels_path, 'labels.pkl'), en_pickle=True)
    train_sample_map=pd.read_csv(osp.join(labels_path, 'train_sample_map.tsv'), sep="\t")
    n_dim=list(labels_all.values())[0].shape[-1] # dim of labels stored, if 2 dim then lat/long
    pop_arr_train = repeat_pop_arr(train_sample_map)
    labelsTrainArr=np.zeros((len(pop_arr_train), n_dim))
    for i,j in enumerate(pop_arr_train[:,1]):
        labelsTrainArr[i,:]=labels_all.get(j) #5930x3 #samplesxdim
    if n_dim==2: # not n vectors but lat/long
        lat=labelsTrainArr[..., 0]
        long=labelsTrainArr[..., 1] 
        labelsTrainArr=convert_nVector(lat,long)
    labelsTrainArr=labelsTrainArr[np.newaxis,:,:]
    preds=preds[:,np.newaxis,:]
    preds=np.repeat(preds,labelsTrainArr.shape[0], axis=1) #(100x605)x5930x3
    if distType=="L2":
        distMatrix=np.sum(np.square(preds-labelsTrainArr), axis=2) #(100x605)x5930
    idx=np.argmin(distMatrix, axis=1)#60500
    mappedSpArr=np.array(pop_arr_train[idx,3]) if labelType=="superpop" else np.array(pop_arr_train[idx,2])
    return mappedSpArr

def getPairwiseDistancePops(**kwargs):
    """
    compute pairwise distance of populations across the windowed
    dimension
    """
    distMetric=kwargs('distMetric')
    ...



