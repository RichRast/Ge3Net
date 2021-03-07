import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from helper_funcs import load_path, save_file
from numpy import linalg as LA
from decorators import timer

POP_ORDER = ['EAS', 'SAS', 'WAS', 'OCE', 'AFR', 'AMR', 'EUR']

def PCA_space(x, n_comp=3, verbose = True):
    pca=decomposition.PCA(n_components=n_comp, whiten=True, random_state=10)
    #x = np.where(x==0,-1,1)
    centered_data = x-(np.sum(x, axis=0)/x.shape[0])
    #Standardized_data=StandardScaler().fit_transform(x)
    pca_data=pca.fit_transform(centered_data)
    print(f'explained_variance_ratio_:{pca.explained_variance_ratio_}')
    return pca_data

@timer
def PCA_space_revised(vcf_snp, idx_lst, n_comp=3, extended_pca = False, pop_arr=None, n_way = 7, n_comp_subclass = 2):
    """
    vcf_snp: entire vcf snp consisting of train, valid and test snps
    idx_list: [train_sample_map, valid_sample_map, test_sample_map]

    This function computes the pca transformation matrix by fitting the 
    vcf snp for train data and computes the projection of valid vcf snp and
    test vcf snp using the computed transformation.
    """
    pca=decomposition.PCA(n_components=n_comp, random_state=10)
    vcf_train = vcf_snp[idx_lst[0]]
    vcf_valid = vcf_snp[idx_lst[1]]
    vcf_test = vcf_snp[idx_lst[2]]
    mean_train = np.sum(vcf_train, axis=0)/vcf_train.shape[0]
    std_train = vcf_train.std(axis=0)
    std_train[std_train==0] = 1

    centered_train = (vcf_train - mean_train)/std_train
    centered_valid = (vcf_valid - mean_train)/std_train
    centered_test = (vcf_test - mean_train)/std_train

    # centered_train = vcf_train - mean_train
    # centered_valid = vcf_valid - mean_train
    # centered_test = vcf_test - mean_train
   
    pca_train = pca.fit(centered_train)
    PCA_labels_train = pca_train.transform(centered_train)
    PCA_labels_valid = pca_train.transform(centered_valid)
    PCA_labels_test = pca_train.transform(centered_test)
    print(f'explained_variance_ratio_:{pca.explained_variance_ratio_}')
    print(f'explained_variance_:{pca.explained_variance_}')
    print(f'cumulative variance  {np.round( pca.explained_variance_.cumsum()[-1],2)}, and variance ratio \
         {np.round( pca.explained_variance_ratio_.cumsum()[-1]*100, 2)}%')

    if extended_pca:
        pca_train_subclass = {}
        pca_subclass = decomposition.PCA(n_components=n_comp_subclass, random_state=10)
        for i in range(n_way):
            idx_subclass = np.nonzero(pop_arr==i)[0]
            pca_train_subclass[i] = pca_subclass.fit(centered_train[idx_subclass])
            print(f'explained_variance_ratio for subclass {i} :{pca_train_subclass[i].explained_variance_ratio_}')
            print(f'explained_variance_ for subclass {i} :{pca_train_subclass[i].explained_variance_}')
            print(f'cumulative variance  {np.round( pca_train_subclass[i].explained_variance_.cumsum()[-1],2)}, and variance ratio \
            {np.round( pca_train_subclass[i].explained_variance_ratio_.cumsum()[-1]*100, 2)}%')
            if i >0:
                PCA_labels_train_subclass = np.hstack((PCA_labels_train_subclass, pca_train_subclass[i].transform(centered_train)))
                PCA_labels_valid_subclass = np.hstack((PCA_labels_valid_subclass, pca_train_subclass[i].transform(centered_valid)))
                PCA_labels_test_subclass = np.hstack((PCA_labels_test_subclass, pca_train_subclass[i].transform(centered_test))) 
            else:
                PCA_labels_train_subclass = pca_train_subclass[i].transform(centered_train)
                PCA_labels_valid_subclass = pca_train_subclass[i].transform(centered_valid)
                PCA_labels_test_subclass = pca_train_subclass[i].transform(centered_test)    
        
        PCA_labels_train = np.hstack((PCA_labels_train, PCA_labels_train_subclass))
        PCA_labels_valid = np.hstack((PCA_labels_valid, PCA_labels_valid_subclass))
        PCA_labels_test = np.hstack((PCA_labels_test, PCA_labels_test_subclass))
        pca_train = [pca_train, pca_train_subclass]

    return PCA_labels_train, PCA_labels_valid, PCA_labels_test, pca_train

@timer
def PCA_space_residual(vcf_snp, idx_lst, n_comp=44, n_comp_overall=3, extended_pca = False, pop_arr=None, n_way = 7, n_comp_subclass = 2):
    """
    vcf_snp: entire vcf snp consisting of train, valid and test snps
    idx_list: [train_sample_map, valid_sample_map, test_sample_map]

    This function computes the pca transformation matrix by fitting the 
    vcf snp for train data and computes the projection of valid vcf snp and
    test vcf snp using the computed transformation.
    """
    pca=decomposition.PCA(n_components=n_comp, random_state=10)
    [vcf_train, vcf_valid, vcf_test] = [vcf_snp[idx_lst[i]] for i in range(len(idx_lst))]
    mean_train = np.sum(vcf_train, axis=0)/vcf_train.shape[0]
    std_train = vcf_train.std(axis=0)
    std_train[std_train==0] = 1

    norm_train = (vcf_train - mean_train)/std_train
    norm_valid = (vcf_valid - mean_train)/std_train
    norm_test = (vcf_test - mean_train)/std_train
   
    pca_train = pca.fit(norm_train)
    PCA_transform_train = pca_train.transform(norm_train)
    PCA_transform_valid = pca_train.transform(norm_valid)
    PCA_transform_test = pca_train.transform(norm_test)
    print(f'explained_variance_ratio_:{pca.explained_variance_ratio_[0:10]}')
    print(f'cumulative variance  {np.round( pca.explained_variance_.cumsum()[-1],2)}, and variance ratio \
         {np.round( pca.explained_variance_ratio_.cumsum()[-1]*100, 2)}%')

    if extended_pca:
        pca_train_subclass = {}
        #pop_num = [0,1,2,3,4,5,6]
        #pop_num = [4,2,1,6,0,3,5]
        pop_num = [4,6,2,1,0,3,5]
        #revised_pop_order = 
        n_components = n_comp-n_comp_overall
        n_comp_subclass = 0
        for i,j in enumerate(pop_num):
            n_components -= n_comp_subclass
            pca_subclass = decomposition.PCA(n_components=n_components, random_state=10)
            if i >0:
                subclass_train, subclass_valid, subclass_test = PCA_transform_train_subclass[:,n_comp_subclass:], \
                    PCA_transform_valid_subclass[:,n_comp_subclass:], PCA_transform_test_subclass[:,n_comp_subclass:]
            else:
                subclass_train, subclass_valid, subclass_test = PCA_transform_train[:,n_comp_overall:], \
                    PCA_transform_valid[:,n_comp_overall:], PCA_transform_test[:,n_comp_overall:]
            norm_residual_train = subclass_train
            norm_residual_valid = subclass_valid
            norm_residual_test = subclass_test
            idx_subclass = np.nonzero(pop_arr==j)[0]
            print(f' n_components :{n_components}, norm_residual_train: {norm_residual_train.shape}, \
                number of samples of class {POP_ORDER[j]} : {len(idx_subclass)}')
            pca_train_subclass[i] = pca_subclass.fit(norm_residual_train[idx_subclass])
            print(f'explained_variance_ratio for subclass {POP_ORDER[j]} :{pca_train_subclass[i].explained_variance_ratio_}')
            print(f'cumulative variance  {np.round( pca_train_subclass[i].explained_variance_.cumsum()[-1],2)}, and variance ratio \
            {np.round( pca_train_subclass[i].explained_variance_ratio_.cumsum()[-1]*100, 2)}%')
            var_choose = [i for i in pca_train_subclass[i].explained_variance_ratio_ if i >0.05]
            n_comp_subclass = len(var_choose)
            print(f'n_comp_subclass for subclass {POP_ORDER[j]} : {n_comp_subclass}')
            PCA_transform_train_subclass = pca_train_subclass[i].transform(norm_residual_train)
            PCA_transform_valid_subclass = pca_train_subclass[i].transform(norm_residual_valid)
            PCA_transform_test_subclass = pca_train_subclass[i].transform(norm_residual_test)

            if i >0:
                PCA_labels_train_subclass = np.hstack((PCA_labels_train_subclass, PCA_transform_train_subclass[:,0:n_comp_subclass]))
                PCA_labels_valid_subclass = np.hstack((PCA_labels_valid_subclass, PCA_transform_valid_subclass[:,0:n_comp_subclass]))
                PCA_labels_test_subclass = np.hstack((PCA_labels_test_subclass, PCA_transform_test_subclass[:,0:n_comp_subclass])) 
            else:
                PCA_labels_train_subclass = pca_train_subclass[i].transform(norm_residual_train)[:,0:n_comp_subclass]
                PCA_labels_valid_subclass = pca_train_subclass[i].transform(norm_residual_valid)[:,0:n_comp_subclass]
                PCA_labels_test_subclass = pca_train_subclass[i].transform(norm_residual_test)[:,0:n_comp_subclass]    
        
        PCA_labels_train = np.hstack((PCA_transform_train[:,0:n_comp_overall], PCA_labels_train_subclass))
        PCA_labels_valid = np.hstack((PCA_transform_valid[:,0:n_comp_overall], PCA_labels_valid_subclass))
        PCA_labels_test = np.hstack((PCA_transform_test[:,0:n_comp_overall], PCA_labels_test_subclass))
        pca_train = [pca_train, pca_train_subclass]

    print(f'PCA_labels_train shape: {PCA_labels_train.shape}')
    return PCA_labels_train, PCA_labels_valid, PCA_labels_test, pca_train

def get_hamming_matrix(x):
    Hamming_matrix = scipy.spatial.distance.cdist(x,x, metric="hamming")
    return Hamming_matrix

def get_eucledian_matrix(x):
    Eucledian_matrix = scipy.spatial.distance.cdist(x,x, metric="euclidean")
    return Eucledian_matrix

def compute_MDS(dissimilarity_matrix):
    dissimilarity_square = np.square(dissimilarity_matrix)
    n = dissimilarity_matrix.shape[0]
    centering_matrix = np.eye(n)-(1/n)*np.ones((n,n))
    w, v = LA.eig(-0.5*np.dot(np.dot(centering_matrix, dissimilarity_square),(centering_matrix)))
    return (w.T)*v

@timer
def MDS_space(x, distance_matrix, data_folder, verbose = True):
    
    if distance_matrix=="hamming_saved":
        path = data_folder + "/dissimilarity_matrix.npy"
        dissimilarity_matrix = load_path(path)
    elif distance_matrix=="eucledian_saved":
        path = data_folder + "/euc_dissimilarity_matrix.npy"
        dissimilarity_matrix = load_path(path)
    elif distance_matrix=="hamming":
        dissimilarity_matrix = get_hamming_matrix(x)
        if verbose:
            print(f"hamming matrix computed with shape :{dissimilarity_matrix.shape}")
        path = data_folder + "/dissimilarity_matrix.npy"
        save_file(path, dissimilarity_matrix)
    elif distance_matrix=="eucledian":
        dissimilarity_matrix = get_eucledian_matrix(x)
        if verbose:
            print(f"eucledian matrix computed with shape :{dissimilarity_matrix.shape}")
        path = data_folder + "/euc_dissimilarity_matrix.npy"
        save_file(path, dissimilarity_matrix)
              
    X_transformed = compute_MDS(dissimilarity_matrix)
    
    if verbose:
        print(f"MDS embeddings shape:{X_transformed.shape}")
    
    path = data_folder + "/MDS_train_founders.npy"
    save_file(path, X_transformed)
    
    return X_transformed, dissimilarity_matrix