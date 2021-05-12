from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import seaborn as sns
import scipy
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import os
import os.path as osp
import subprocess
from abc import ABC, abstractmethod
from src.utils.dataUtil import load_path, save_file, getValueBySelection
from src.utils.decorators import timer
import umap
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from numpy import linalg as LA

class UnsupervisedSpace(ABC):
    """
    Abstract class for creating an unsupervised space
    """
    @abstractmethod
    def __init__(self, n_comp, data_out_path, pop_order, random_state):
        self.n_comp=n_comp
        self.random_state=random_state
        self.data_out_path=data_out_path
        self.pop_order = pop_order
        self.reducer = None
        self.name = None
        
    @timer
    def fit(self, X):
        self.reducer.fit(X)
    
    @timer
    def transform(self, X):
        self.labels = self.reducer.transform(X)

    @timer
    def fit_transform(self, X):
        self.labels = self.reducer.fit_transform(X)
        mean = self.labels.mean(axis=0)
        std = self.labels.std(axis=0)
        self.labels = (self.labels-mean)/std
 
    def _plot3dEmbeddings(self, pop_arr, random_idx, rev_pop_dict):
        fig, ax = plt.subplots(figsize=(10,12))

        colors_pop = sns.color_palette("rainbow", len(self.pop_order))
        n_way=len(self.pop_order)
        
        ax = plt.subplot(111, projection='3d')
        for i in range(n_way):
            idx_label = np.nonzero(pop_arr[:,3]==i)[0]
            ax.scatter(self.labels[idx_label,0], self.labels[idx_label,1], self.labels[idx_label,2], s=10,\
                    color=colors_pop[i] , label = self.pop_order[i])
            
            handles, labels = ax.get_legend_handles_labels()
            lgnd = ax.legend(handles, labels, bbox_to_anchor=(0.9,0.5+(i/20)))
            for l in lgnd.legendHandles:
                l._sizes = [30]
            for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
                # make the grid lines transparent
                a.set_pane_color((1.0, 1.0, 1.0, 0.0))
                a._axinfo["grid"]['color'] =  (1,1,1,0)
            
        for k in random_idx:
            ax.text(self.labels[k,0], self.labels[k,1], self.labels[k,2], \
                    s = rev_pop_dict[pop_arr[k,2]],\
                fontweight='bold', fontsize = 12)
        return lgnd, fig

    def _plot2dEmbeddings(self, pop_arr, random_idx, rev_pop_dict):    
        fig, ax = plt.subplots(figsize=(10,12))

        colors_pop = sns.color_palette("rainbow", len(self.pop_order))
        n_way=len(self.pop_order)     
        for i in range(n_way):
            idx_label = np.nonzero(pop_arr[:,3]==i)[0]
            ax.scatter(self.labels[idx_label,0], self.labels[idx_label,1], s=5, \
                    color=colors_pop[i] , label = self.pop_order[i] )

            handles, labels = ax.get_legend_handles_labels()
            lgnd = ax.legend(handles, labels, bbox_to_anchor=(1.0,0.8))
            for l in lgnd.legendHandles:
                l._sizes = [30]

        for k in random_idx:
            ax.text(self.labels[k,0], self.labels[k,1], \
                    s = rev_pop_dict[pop_arr[k,2]],\
                fontweight='bold', fontsize = 12)
        return lgnd, fig
    
    def plot_embeddings(self, rev_pop_dict, pop_arr):
        # print randomly 30 granular pop on the PCA plot 
        random_idx = np.random.choice(self.labels.shape[0], 20)
        if self.n_comp>=3:
            lgd, fig1 = self._plot3dEmbeddings(pop_arr, \
            random_idx, rev_pop_dict)
        else:
            lgd, fig1 = self._plot2dEmbeddings(pop_arr, \
            random_idx, rev_pop_dict)
        plt.show()
        fig1.savefig(osp.join(self.data_out_path, 'labels.png'), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close('all')

    def saveLabels(self, pop_arr):
        # form the dict with vcf_idx and embeddings
        vcf_idx = list(pop_arr[:,1])
        self.lbls_dict = {k:v for k,v in zip(vcf_idx, self.labels)}
        save_file(osp.join(self.data_out_path, 'labels.pkl'), self.lbls_dict, en_pickle=True)
        labelsBySample = pd.DataFrame(columns=['Sample', 'ref_idx', 'labels'])
        labelsBySample['ref_idx']=list(self.lbls_dict.keys())
        labelsBySample['labels']=list(self.lbls_dict.values())
        labelsBySample['Sample']=labelsBySample['ref_idx'].apply(lambda x: getValueBySelection(pop_arr, 1, x, 0) )
        labelsBySample.to_csv(osp.join(self.data_out_path, 'labelsBySample.tsv'), sep="\t", index=None)
    
    def __call__(self, X, rev_pop_dict, pop_arr):
        self.fit_transform(X)
        self.plot_embeddings(rev_pop_dict, pop_arr)
        self.saveLabels(pop_arr)        

class pcaSpace(UnsupervisedSpace):
    def __init__(self, n_comp, data_out_path, pop_order, random_state, whiten=True):
        super().__init__(n_comp, data_out_path, pop_order, random_state)
        self.reducer = decomposition.PCA(n_components=self.n_comp, \
            random_state=self.random_state, whiten=whiten)
        self.name = "PCA"

    @timer
    def fit(self, X):
        super().fit(X)
        print(f'explained_variance_ratio_:{self.reducer.explained_variance_ratio_}')
        print(f'cumulative variance  {np.round( self.reducer.explained_variance_.cumsum()[-1],2)}, and variance ratio \
            {np.round( self.reducer.explained_variance_ratio_.cumsum()[-1]*100, 2)}%')
        self.eigenvalues = self.reducer.explained_variance_
        self.singularValues = self.reducer.singular_values_

    @timer    
    def transform(self, X):
        super().transform(X)
        # eigenvecs (singular vecs across samples)
        # self.labels /= self.singularValues

    @timer
    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)

class sPca(pcaSpace):
        pass

class residualPca(pcaSpace):
    def __init__(self, n_comp, data_out_path, n_comp_overall, n_comp_subclass, \
        pop_arr, pop_order, random_state, whiten=False):
        super().__init__(n_comp, data_out_path, random_state, whiten)
        self.n_comp_overall=n_comp_overall
        self.n_comp_subclass=n_comp_subclass
        self.pop_arr=pop_arr
        self.pop_order=pop_order
        self.name="residual_PCA"

    def _getSubclassName(self, pop_num_val):
        if isinstance(pop_num_val, list):
            return "_".join([str(self.pop_order[j]) for j in pop_num_val])
        else:
            return self.pop_order[pop_num_val]

    @timer
    def fit_transform(self, X):
        self.pca_overall=pcaSpace(self.n_comp, self.data_out_path, 10)
        self.pca_overall.fit(X)
        self.pca_overall.transform(X)
        self.pca_subclass = {}
        pop_num = np.arange(len(self.pop_order))
        n_components = self.n_comp-self.n_comp_overall
        subclass_labels=[]
        for i, pop_num_val in enumerate(pop_num):
            if i >0:
                subclass_X = self.pca_subclass[i-1].labels[:,self.n_comp_subclass:]
            else:
                subclass_X= self.pca_overall.labels[:,self.n_comp_overall:]
            self.pca_subclass[i] = pcaSpace(n_components, self.data_out_path, 10)
            self.pca_subclass[i].name = self._getSubclassName(pop_num_val)
            self.pca_subclass[i].fit(subclass_X)
            self.pca_subclass[i].transform(subclass_X)

            n_components -= self.n_comp_subclass
            subclass_labels.append(self.pca_subclass[i].labels)

        self.labels = np.hstack((self.pca_overall.labels, subclass_labels))

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def plot_embeddings(self, rev_pop_dict):
        lgnd, fig = self.pca_overall._plot3dEmbeddings(rev_pop_dict, self.pop_arr, self.pop_order)
        plt.title(f" Overall PCA")
        plt.show()
        fig.savefig(osp.join(self.data_out_path, f'labels.png'), bbox_extra_artists=(lgnd,), bbox_inches='tight')
        pop_num = np.arange(len(self.pop_order))
        for j, _ in enumerate(pop_num):
            # plot all the classes for the same subclass for train
            # randomly select 30 granular pops for the particular subpop                
            random_idx = np.random.choice(self.labels.shape[0], 20)
            lgnd, fig = self.pca_subclass[j]._plot2dEmbeddings(self.pop_arr, random_idx, rev_pop_dict, self.pop_order)
            plt.title(f" subclass : {self.pca_subclass[j].name}")
            plt.show()
            fig.savefig(osp.join(self.data_out_path, f'labels_{self.pca_subclass[j].name}.png'), bbox_extra_artists=(lgnd,), bbox_inches='tight')
        plt.close('all')

    def __call__(self, X, rev_pop_dict, pop_arr):
        self.fit_transform(X)
        self.plot_embeddings(rev_pop_dict)
        self.saveLabels(pop_arr)
                    
class umapSpace(UnsupervisedSpace):
    def __init__(self, n_comp, data_out_path, pop_order, random_state):
        super().__init__(n_comp, data_out_path, pop_order, random_state)
        self.reducer=umap.UMAP(n_components=self.n_comp, random_state=self.random_state)
        self.name = "Umap"

class tsneSpace(UnsupervisedSpace):
    def __init__(self, n_comp, data_out_path, pop_order, random_state):
        super().__init__(n_comp, data_out_path, pop_order, random_state)
        self.reducer=TSNE(n_components=n_comp, random_state=random_state)
        self.name = "TSNE"
    
class spectralEmbeddingSpace(UnsupervisedSpace):
    def __init__(self, n_comp, data_out_path, pop_order, random_state):
        super().__init__(n_comp, data_out_path, pop_order, random_state)
        self.reducer=SpectralEmbedding(n_components=n_comp, random_state=random_state)
        self.name = "SpectralEmbedding"

class thinningPcaSpace(UnsupervisedSpace):
    def __init__(self, data_out_path, pop_sample_map):
        self.data_out_path=data_out_path
        self.name = "Thinning_PCA_plink"
        self.pop_sample_map = pop_sample_map

    def fit_transform(self, sh_args, plink_path):
        assert isinstance(sh_args, list), "Invalid args passed for plink pca"
        os.chdir(os.environ.get('USER_PATH'))
        print("Executing command : \n")
        print(' '.join(["/Batch_job_scripts/plink_pca.sh", *sh_args ]))
        subprocess.check_call(' '.join(["./Batch_job_scripts/plink_pca.sh", *sh_args ]), shell=True)
        eigenvecs = pd.read_csv(osp.join(plink_path, 'plinkpca.eigenvec'), sep="\t")
        self.labels  = self.pop_sample_map.merge(eigenvecs, how="inner", \
            left_on="Sample", right_on="IID").reset_index(drop=True)
        
    def fit(self, X):   
        raise NotImplementedError

    def transform(self, X):   
        raise NotImplementedError

    def plot_embeddings(self):
        random_idx= np.random.choice(len(self.labels), 30)
        pca_labels=self.labels[['PC1', 'PC2', 'PC3']].values
        pca_superpop=self.labels[['Superpopulation code']].values
        pca_granularpop=self.labels[['Population']].values
        fig, ax = plot3dEmbeddingsGeneric(pca_labels, pca_superpop, pca_granularpop, random_idx)
        plt.title("Thinning PCA whole genome")
        plt.show()
        fig.savefig(osp.join(self.data_out_path, f'labels.png'), bbox_inches='tight')
        plt.close('all')

    def saveLabels(self):
        self.labels.to_csv(osp.join(self.data_out_path, 'labelsBySample.tsv'), sep="\t", index=None)
    
    def __call__(self, sh_args, plink_path):
        self.fit_transform(sh_args, plink_path)
        self.plot_embeddings()
        self.saveLabels()
    
# **** Other generic functions below *******

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

def plot3dEmbeddingsGeneric(labels, superpop, granular_pop, random_idx):
    fig, ax = plt.subplots(figsize=(12,10))
    ax= Axes3D(fig)
    superpop_num = superpop.unique()
    superpop_dict={k:v for k,v in zip(superpop_num, np.arange(len(superpop_num)))}
    colors_pop = sns.color_palette("rainbow", len(superpop_num))
    ax.scatter(labels[:,0], labels[:,1], labels[:,2], \
            color=[colors_pop[superpop_dict[x]] for x in superpop], s=50)
    for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
            # make the grid lines transparent
            a.set_pane_color((1.0, 1.0, 1.0, 0.0))
            a._axinfo["grid"]['color'] =  (1,1,1,0)

    patches=[]
    for i,j in enumerate(superpop_num):
        patches.append(mpatches.Patch(color = colors_pop[i], label = j))
    ax.legend(handles=patches, bbox_to_anchor=(0.3,0.85,0.7,0),loc=5,ncol=4)

    for k in random_idx:
        ax.text(labels[k,0], labels[k,1], labels[k,2], \
                s = granular_pop[k],\
            fontweight='bold', fontsize = 12)
    return fig, ax