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

def PCA_space(x, n_comp=3, verbose = True):
    pca=decomposition.PCA(n_components=n_comp, whiten=True, random_state=10)
    #x = np.where(x==0,-1,1)
    centered_data = x-(np.sum(x, axis=0)/x.shape[0])
    #Standardized_data=StandardScaler().fit_transform(x)
    pca_data=pca.fit_transform(centered_data)
    print(f'explained_variance_ratio_:{pca.explained_variance_ratio_}')
    return pca_data

@timer
def PCA_space_revised(vcf_snp, idx_lst, n_comp=3, verbose = True):
    """
    vcf_snp: entire vcf snp consisting of train, valid and test snps
    idx_list: [train_sample_map, valid_sample_map, test_sample_map]

    This function computes the pca transformation matrix by fitting the 
    vcf snp for train data and computes the projection of valid vcf snp and
    test vcf snp using the computed transformation.
    """
    pca=decomposition.PCA(n_components=n_comp, whiten=True, random_state=10)
    vcf_train = vcf_snp[idx_lst[0]]
    vcf_valid = vcf_snp[idx_lst[1]]
    vcf_test = vcf_snp[idx_lst[2]]
    mean_train = np.sum(vcf_train, axis=0)/vcf_train.shape[0]
    std_train = vcf_train.std(axis=0)
    std_train[std_train==0] = 1

    # centered_train = (vcf_train - mean_train)/std_train
    # centered_valid = (vcf_valid - mean_train)/std_train
    # centered_test = (vcf_test - mean_train)/std_train

    centered_train = vcf_train - mean_train
    centered_valid = vcf_valid - mean_train
    centered_test = vcf_test - mean_train
   
    pca_train = pca.fit(centered_train)
    PCA_labels_train = pca_train.transform(centered_train)
    PCA_labels_valid = pca_train.transform(centered_valid)
    PCA_labels_test = pca_train.transform(centered_test)
    print(f'explained_variance_ratio_:{pca.explained_variance_ratio_}')
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