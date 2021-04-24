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
import umap
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE

# POP_ORDER = ['EAS', 'SAS', 'WAS', 'OCE', 'AFR', 'AMR', 'EUR']

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
    print(f"vcf_train shape:{vcf_train.shape}")
    vcf_valid = vcf_snp[idx_lst[1]]
    print(f"vcf_valid shape:{vcf_valid.shape}")
    vcf_test = vcf_snp[idx_lst[2]]
    print(f"vcf_test shape:{vcf_test.shape}")
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
def PCA_space_residual(vcf_snp, idx_lst, pop_order, n_comp=44, n_comp_overall=3, extended_pca = False, pop_arr=None, n_way = 7, n_comp_subclass = 2):
    """
    vcf_snp: entire vcf snp consisting of train, valid and test snps
    idx_list: [train_sample_map, valid_sample_map, test_sample_map]

    This function computes the pca transformation matrix by fitting the 
    vcf snp for train data and computes the projection of valid vcf snp and
    test vcf snp using the computed transformation.
    """
    pca=decomposition.PCA(n_components=n_comp, whiten=True, random_state=10)
    [vcf_train, vcf_valid, vcf_test] = [vcf_snp[idx_lst[i]] for i in range(len(idx_lst))]
    mean_train = np.sum(vcf_train, axis=0)/vcf_train.shape[0]
    std_train = vcf_train.std(axis=0)
    std_train[std_train==0] = 1

    # norm_train = (vcf_train - mean_train)/std_train
    # norm_valid = (vcf_valid - mean_train)/std_train
    # norm_test = (vcf_test - mean_train)/std_train

    norm_train = (vcf_train - mean_train)
    norm_valid = (vcf_valid - mean_train)
    norm_test = (vcf_test - mean_train)
   
    pca_train = pca.fit(norm_train)
    PCA_transform_train = pca_train.transform(norm_train)
    PCA_transform_valid = pca_train.transform(norm_valid)
    PCA_transform_test = pca_train.transform(norm_test)
    print(f'explained_variance_ratio_:{pca.explained_variance_ratio_[0:10]}')
    print(f'cumulative variance  {np.round( pca.explained_variance_.cumsum()[-1],2)}, and variance ratio \
         {np.round( pca.explained_variance_ratio_.cumsum()[-1]*100, 2)}%')

    if extended_pca:
        pca_train_subclass = {}
        pop_num = np.arange(len(pop_order))
        # pop_num = [4,6,2,1,0,3,5]
        # pop_num = [4,[6,2,1,0],3,5]
        # pop_num = [4,[6,2,1],0,3,5]
        n_components = n_comp-n_comp_overall
        for i, pop_num_val in enumerate(pop_num):
            pca_subclass = decomposition.PCA(n_components=n_components, whiten=True, random_state=10)
            if i >0:
                subclass_train, subclass_valid, subclass_test = PCA_transform_train_subclass[:,n_comp_subclass:], \
                    PCA_transform_valid_subclass[:,n_comp_subclass:], PCA_transform_test_subclass[:,n_comp_subclass:]
            else:
                subclass_train, subclass_valid, subclass_test = PCA_transform_train[:,n_comp_overall:], \
                    PCA_transform_valid[:,n_comp_overall:], PCA_transform_test[:,n_comp_overall:]
            norm_residual_train = subclass_train
            norm_residual_valid = subclass_valid
            norm_residual_test = subclass_test
            if isinstance(pop_num_val, list):
                idx_subclass = np.nonzero(np.isin(pop_arr, pop_num_val))[0]
                tmp_pop_name = "_".join([str(pop_order[j]) for j in pop_num_val])
                print(f'mask array, idx_subclass, subclass: {np.isin(pop_arr, pop_num_val)}, {idx_subclass}, {tmp_pop_name}')
            else:    
                idx_subclass = np.nonzero(pop_arr==pop_num_val)[0]
                tmp_pop_name = pop_order[pop_num_val] 
            print(f' n_components :{n_components}, norm_residual_train: {norm_residual_train.shape}, \
                number of samples of class {tmp_pop_name} : {len(idx_subclass)}')
            pca_train_subclass[i] = pca_subclass.fit(norm_residual_train[idx_subclass])
            print(f'explained_variance_ratio for subclass {tmp_pop_name} :{pca_train_subclass[i].explained_variance_ratio_}')
            print(f'cumulative variance  {np.round( pca_train_subclass[i].explained_variance_.cumsum()[-1],2)}, and variance ratio \
            {np.round( pca_train_subclass[i].explained_variance_ratio_.cumsum()[-1]*100, 2)}%')
            print(f'n_comp_subclass for subclass {tmp_pop_name} : {n_comp_subclass}')
            PCA_transform_train_subclass = pca_train_subclass[i].transform(norm_residual_train)
            PCA_transform_valid_subclass = pca_train_subclass[i].transform(norm_residual_valid)
            PCA_transform_test_subclass = pca_train_subclass[i].transform(norm_residual_test)
            n_components -= n_comp_subclass
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
    else:
        PCA_labels_train = PCA_transform_train
        PCA_labels_valid = PCA_transform_valid
        PCA_labels_test = PCA_transform_test

    print(f'PCA_labels_train shape: {PCA_labels_train.shape}')
    return PCA_labels_train, PCA_labels_valid, PCA_labels_test, pca_train

@timer
def PCA_space_extended(vcf_snp, idx_lst, pop_order, n_comp=44, n_comp_overall=3, extended_pca = False, pop_arr=None, n_way = 7, n_comp_subclass = 2):
    """
    vcf_snp: entire vcf snp consisting of train, valid and test snps
    idx_list: [train_sample_map, valid_sample_map, test_sample_map]

    This function computes the pca transformation matrix by fitting the 
    vcf snp for train data and computes the projection of valid vcf snp and
    test vcf snp using the computed transformation.
    """
    pca=decomposition.PCA(n_components=n_comp_overall, whiten=True, random_state=10)
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
        # pop_num = [0,1,2,3,4,5,6]
        # pop_num = [4,2,1,6,0,3,5]
        pop_num = np.arange(len(pop_order))
        # pop_num = [4,6,2,1,0,3,5]
        for i,j in enumerate(pop_num):
            pca_subclass = decomposition.PCA(n_components=n_comp_subclass, whiten=True, random_state=10)
            norm_residual_train = norm_train
            norm_residual_valid = norm_valid
            norm_residual_test = norm_test
            idx_subclass = np.nonzero(pop_arr==j)[0]
            print(f' number of samples of class {pop_order[j]} : {len(idx_subclass)}')
            pca_train_subclass[i] = pca_subclass.fit(norm_residual_train[idx_subclass])
            print(f'explained_variance_ratio for subclass {pop_order[j]} :{pca_train_subclass[i].explained_variance_ratio_}')
            print(f'cumulative variance  {np.round( pca_train_subclass[i].explained_variance_.cumsum()[-1],2)}, and variance ratio \
            {np.round( pca_train_subclass[i].explained_variance_ratio_.cumsum()[-1]*100, 2)}%')
            
            PCA_transform_train_subclass = pca_train_subclass[i].transform(norm_residual_train)
            PCA_transform_valid_subclass = pca_train_subclass[i].transform(norm_residual_valid)
            PCA_transform_test_subclass = pca_train_subclass[i].transform(norm_residual_test)

            if i >0:
                PCA_labels_train_subclass = np.hstack((PCA_labels_train_subclass, PCA_transform_train_subclass))
                PCA_labels_valid_subclass = np.hstack((PCA_labels_valid_subclass, PCA_transform_valid_subclass))
                PCA_labels_test_subclass = np.hstack((PCA_labels_test_subclass, PCA_transform_test_subclass)) 
            else:
                PCA_labels_train_subclass = PCA_transform_train_subclass
                PCA_labels_valid_subclass = PCA_transform_valid_subclass
                PCA_labels_test_subclass = PCA_transform_test_subclass
        
        PCA_labels_train = np.hstack((PCA_transform_train, PCA_labels_train_subclass))
        PCA_labels_valid = np.hstack((PCA_transform_valid, PCA_labels_valid_subclass))
        PCA_labels_test = np.hstack((PCA_transform_test, PCA_labels_test_subclass))
        pca_train = [pca_train, pca_train_subclass]

    print(f'PCA_labels_train shape: {PCA_labels_train.shape}')
    return PCA_labels_train, PCA_labels_valid, PCA_labels_test, pca_train

def sPCA(vcf_snp, idx_lst, pop_order, n_comp, n_comp_overall):
    pca=decomposition.PCA(n_components=n_comp, whiten=True, random_state=10)
    [vcf_train, vcf_valid, vcf_test] = [vcf_snp[idx_lst[i]] for i in range(len(idx_lst))]
    mean_train = np.sum(vcf_train, axis=0)/vcf_train.shape[0]
    std_train = vcf_train.std(axis=0)
    std_train[std_train==0] = 1
    # pearson correlation coeficiant for each pair of snps
    


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

def Umap_space(X, n_comp):
    reducer = umap.UMAP(random_state=42)
    reducer.fit(X)
    embedding = reducer.transform(X)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    print(f"embedding shape :{embedding.shape}")

def TSNE_space(X, n_comp):
    embedding = TSNE(n_components = n_comp)
    X_transformed = embedding.fit_transform(X)
    print(f"X_transformed shape: {X_transformed.shape}")
