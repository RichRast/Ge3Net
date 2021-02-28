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