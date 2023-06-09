import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib import ticker
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from src.utils.modelUtil import convert_coordinates
from src.utils.dataUtil import getValueBySelection
from src.main.evaluation import cpMethod

def plot_coordinates_map(granularPopNames, pred_coord, lbl_coord, **kwargs):
    """
    plotly plot for coordinates on a map
    label: target label vector (int) for the specific sample
    data_coordinates: lat, long for the specific sample
    rev_pop_order: dict with keys as target label ints and values
    as granular population name
    """
    cpShow=kwargs.get('cpShow')
    pred_cps=kwargs.get('pred_cps') 
    alpha=[1.0]*len(lbl_coord)
    if cpShow:
        unknownIdx=np.nonzero(pred_cps)[0]
        granularPopNames=["UNK" if i in unknownIdx else granularPopNames[i] for i in range(len(granularPopNames))]
    fig = go.Figure(go.Scattergeo())
    colors_pop = sns.color_palette("rainbow", len(np.unique(granularPopNames)))
    
    colors_pop_dict = {k:v for k,v in zip(np.unique(granularPopNames), colors_pop)}
    colors_pop_dict['UNK']=(188,188,188) # grey color
    fig.add_trace(go.Scattergeo(lon=pred_coord[:,1], lat=pred_coord[:,0], text = granularPopNames,\
    marker_color=['rgba'+str(colors_pop_dict[x]+(y,)) for x,y in zip(granularPopNames, alpha)]))
    fig.update_traces(marker_size = 5)
    fig.add_trace(go.Scattergeo(lon=lbl_coord[:,1], lat=lbl_coord[:,0], marker = dict(symbol = 'square'), \
                                text = granularPopNames))
    fig.update_traces(marker_size = 5)

    fig.show()
    plt.show()
    plt.close('all')


def plot_sample(granularPopSample, y_predSample, y_trueSample, superop_dict, chmlen, **kwargs):
    backgroundAxis=kwargs.get('backgroundAxis')
    y_predCps=kwargs.get('y_predCps')
    y_preds=kwargs.get('y_preds')
    superpops=kwargs.get('superpops')
    cpShow=kwargs.get('cpShow')
    pred_cps=kwargs.get('pred_cps')
    mappedSpArr=kwargs.get('mappedSpArr')
    fig, ax = plt.subplots(figsize=(12,14))
    
    gs1 = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[13,1,1])
    ax1=fig.add_subplot(gs1[0],projection='3d')
    ax2=fig.add_subplot(gs1[1])
    ax3=fig.add_subplot(gs1[2])
    for axis in [ax, ax2, ax3]:
        axis.set_yticks([])
        axis.spines['top'].set_color('none')
        axis.spines['left'].set_color('none')
        axis.spines['right'].set_color('none')
    ax.set_xticks([])
    plt.subplots_adjust(hspace=0.01)
    
    if backgroundAxis is not None:
        lgnd, colorsPop_sp_dict =plot_all(ax1, y_preds, superpops, cpShow=False)
    else:
        continentaPops=list(superop_dict.values())
        colors_Sp = sns.color_palette("bright", 10)
        del colors_Sp[1]
        del colors_Sp[4]
        del colors_Sp[5]
        colorsPop_sp_dict = {k:v for k,v in zip(continentaPops, colors_Sp)}
        colorsPop_sp_dict[-1]=(0.7,0.7,0.7) # grey color
        patches=[]
        for k, val in superop_dict.items():
            patches.append(mpatches.Patch(color = colorsPop_sp_dict[val], label = k))
        lgnd = ax1.legend(handles=patches, loc="upper right",fontsize=15)
    ax1.add_artist(lgnd)    
    colors_pop = sns.color_palette("rainbow", len(np.unique(granularPopSample)))
    colors_pop_dict = {k:v for k,v in zip(np.unique(granularPopSample), colors_pop)}

    alpha=[1]*len(y_predSample)
    if cpShow is None:
#         alpha=(pred_cps==0).astype(float)
        unknownIdx=np.nonzero(pred_cps)[0]
        print(f"number of changepoints removed:{len(unknownIdx)},{len(unknownIdx)/len(y_predSample)}")
        granularPopNames=["UNK" if i in unknownIdx else granularPopSample[i] for i in range(len(granularPopSample))]
        colors_pop_dict["UNK"]=(0.9,0.9,0.9) # grey color
        mappedSpArr[unknownIdx]=-1
    
    ax1.scatter(y_predSample[:,0], y_predSample[:,1], y_predSample[:,2], \
               color = [colors_pop_dict[x]+(y,) for x,y in zip(granularPopNames, alpha)], s=50, zorder=0) 
    ax1.scatter(y_trueSample[:,0], y_trueSample[:,1], y_trueSample[:,2], \
               color = [colors_pop_dict[x] for x in granularPopSample], marker='X', s=200, zorder=0)
    
    for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]['color'] = (1,1,1,0)
    patches=[]
    for i, val in enumerate(np.unique(granularPopSample)):
        patches.append(mpatches.Patch(color = colors_pop_dict[val], label = val))
    patches.append(mpatches.Patch(color = (0.9,0.9,0.9), label = "UNK"))
    ax1.legend(handles=patches, loc="upper left", fontsize=20)
    
    #chm plot ground truth
    ax2.scatter(np.arange(len(y_trueSample)),np.ones(len(y_trueSample)),\
                color=[colors_pop_dict[x] for x in granularPopSample], marker='s')
    ax2.set_title('Labeled Chromosome22', fontsize=30, y=0.3)
    #chm plot of predictions
    ax3.scatter(np.arange(len(mappedSpArr)),np.ones(len(mappedSpArr)),\
                color=[colorsPop_sp_dict[x]+(y,) for x,y in zip(mappedSpArr,alpha)], marker='s')
    ax3.set_title('Predicted Chromosome22', fontsize=30, y=0.3)
    
    for ax in [ax2, ax3]:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(4))
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(which='major', width=2, length=10, labelsize=25)
        ax.tick_params(which='minor', width=2, length=10, labelsize=10)
        ax.set_xlim(0, chmlen/1000)
        ax.set_ylim(0.99,1.09)
        positions = [0, chmlen/200, chmlen/1000]
        x_labels = [0, chmlen/2, chmlen]
        ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))
    
    fig.tight_layout()
    plt.show()
    plt.close('all')

def plot_all(ax, y_preds, superpops, superop_dict, **kwargs):
    cpShow=kwargs.get('cpShow')
    pred_cps=kwargs.get('pred_cps')    
    continentaPops=list(superop_dict.values())
    colors_pop = sns.color_palette("bright", 10)
    del colors_pop[1]
    del colors_pop[4]
    del colors_pop[5]
    colors_pop_dict = {k:v for k,v in zip(continentaPops, colors_pop)}
    
    alpha=[0.03]*len(y_preds)
    if cpShow is None:
        alpha=(pred_cps==0).astype(float)
    
    ax.scatter(y_preds[:,0], y_preds[:,1], y_preds[:,2], \
               color = [colors_pop_dict[x]+(y,) for x,y in zip(superpops, alpha)], marker=".", s=0.05,zorder=-1) 
    
    patches=[]
    for k, val in superop_dict.items():
        patches.append(mpatches.Patch(color = colors_pop_dict[val], label = k))
    lgnd = ax.legend(handles=patches, loc="upper right", fontsize=15)
    return lgnd, colors_pop_dict
    
def plot_dist(mean, var , chm):
    """
    plots the distribution (mean and var)
    for a given chm
    """
    fig, (ax1, ax2) = plt.subplots(2)
    plt.rcParams['figure.figsize']=[18,15]
    count, bins, ignored = ax1.hist(mean, 20, density = True)
    ax1.set_title(f"Mean distribution for chm {chm}")
    count, bins, ignored = ax2.hist(var, 10, density = True)
    ax2.set_title(f"variance distribution for chm {chm}")
    plt.show()
    plt.close('all')

def plot_chmBaratBottom(rev_pop_dict):
    # norm = []
    # for i in label_lst:
    #     idx[i]=np.nonzero(true_label_num==i)[0]
    #     norm=colors.LogNorm(vmin=np.percentile(label_gcd[idx[i]],5), vmax=np.percentile(label_gcd[idx[i]],95))
    #     ax1.scatter(np.nonzero(true_label_num==i)[0],np.ones((np.nonzero(true_label_num==i)[0].shape[0])),\
    #                       c = label_gcd[idx[i]], label = inv_dict[i], s=300, marker='s',\
    #                       cmap = colors2[j], \
    #                       norm = norm)
    # ax1.set_yticks([])
    # ax1.set_title('chromosome22', fontsize=55)
    # ax1.spines['top'].set_color('none')
    # ax1.spines['right'].set_color('none')
    # ax1.spines['left'].set_color('none')
    # ax1.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(4))
    # ax1.xaxis.set_ticks_position('bottom')
    # ax1.tick_params(which='major', width=2, length=10, labelsize=55)
    # ax1.tick_params(which='minor', width=2, length=10, labelsize=10)
    # ax1.set_xlim(0, 316)
    # ax1.set_ylim(0.99,1.03)
    # positions = [0, 160, 300]
    # x_labels = [0, 150000, 317410]
    # ax1.xaxis.set_major_locator(ticker.FixedLocator(positions))
    # ax1.xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))
    # plt.show()
    ...

def plot_changepoints(true_cps, pred_cps, y_pred, bocp_rl, y_var, popNames):
    """
    For a single haplotype, plot the comparison of changepoint detection
    by different methods
    """
    fig, ax = plt.subplots(8,1,figsize=[18,30])
    T=len(true_cps)
    ax[0].plot(y_pred)
    ax[0].text(0, np.max(y_pred)-0.5, s=popNames[0], size=15)
    for i in np.nonzero(true_cps)[0]:
        ax[0].plot([i,i], [np.min(y_pred), np.max(y_pred)], 'r' )
        ax[0].text(i, np.max(y_pred)-0.5, s=popNames[i+1], size=15)
    ax[1].plot(pred_cps[cpMethod.gradient.name])
    ax[2].plot(y_var)
    ax[3].plot(pred_cps[cpMethod.mc_dropout.name])
    ax[4].plot(pred_cps[cpMethod.neural_network.name])
    ax[5].plot(np.arange(T+1), bocp_rl)
    ax[6].plot(pred_cps[cpMethod.BOCD.name])
    ax[7].plot(true_cps)
    
    ax[0].set_title("n_vectors")
    ax[1].set_title("Simple gradient(post_process)")
    ax[2].set_title("Mc dropout variance")
    ax[3].set_title("Mc dropout")
    ax[4].set_title("Neural Network predicted cp")
    ax[5].set_title("BOCD (post process) run_length")
    ax[6].set_title("BOCD (post process) cp")
    ax[7].set_title("True Cps")
    plt.show()
    

def chm_plot(label,gcd):
    """
    plot subplots for chm style true label, heat map style chm plot
    of predicted gcd and line plot style chm plot of gcd
    """
    x = np.arange(label.shape[0])
    y1 = np.ones(label.shape[0])
    cmap1 = plt.cm.rainbow
    labels = [inv_dict[i] for i in np.unique(label)]
    colors = ['tab:blue','tab:green','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan' ]
    
    label_lst=np.unique(label)
    fig, (ax1,ax2,ax3) = plt.subplots(nrows = 3, ncols=1, figsize = (12,5))
    
    plt.subplots_adjust(hspace=0.5)
    j=0
    for i in label_lst:
        ax1.scatter(np.nonzero(label==i)[0],np.ones(np.argwhere(label==i).shape[0]),label=inv_dict[i],\
                    cmap=cmap1, c = colors[j])
        j+=1
    ax1.set_yticks([])
    ax1.set_ylabel('True_label')
    
    divider = make_axes_locatable(ax)
    
    ax1.legend(bbox_to_anchor=(1.3,0.9))
    
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('viridis',['yellow','red'])
    im2 = ax2.scatter(x,y1, c = gcd, cmap = cmap2)
    cbar = fig.colorbar(im2, ax=ax2, orientation="horizontal", pad = 0.5)
    cbar.ax.set_xlabel('Km')
    ax2.set_yticks([])
    ax2.set_ylabel('gcd')
    ax3.set_figsize=(12,2)
    ax3.plot(np.arange(gcd.shape[0]),gcd)
    ax3.set_ylabel('gcd Km')
    print(f' Overall gcd : {np.mean(gcd):.3f} Km')
    plt.show()

class Plot_per_epoch(object):
    def __init__(self, n_comp_overall, n_comp_subclass, rev_pop_dict, pop_arr, **kwargs):
        self.n_comp_overall = n_comp_overall
        self.n_comp_subclass = n_comp_subclass
        self.rev_pop_dict = rev_pop_dict
        self.pop_arr = pop_arr
        self.geography=kwargs.get('geography')
        self.pop_order = kwargs.get('pop_order')
        
    def plot_index(self, y_pred_overall, y_target, y_vcf_idx, **kwargs):
        y_pred_subclass=kwargs.get('y_pred_subclass')
        y_pred_sp=kwargs.get('y_pred_sp')
        # for extended pca - plot the overall plot with 6 or 7 subplots for a prediction
        # fig, ax = plt.subplots(len(self.pop_num)+1, figsize=(10,62), gridspec_kw={'height_ratios':[1]+[1]*len(self.pop_num)})
        fig, ax = plt.subplots(figsize=(10,62))
        num_labels_idx = np.unique(y_vcf_idx)
        colors_pop = sns.color_palette("rainbow", len(num_labels_idx))
        colors_pop_dict = {k:v for k,v in zip(num_labels_idx, colors_pop)}
        j =0

        gradient_cp_idx = np.unique(np.where(abs(y_pred_overall[:-1,:]-y_pred_overall[1:,:])>0.3)[0])
            
        # subplot for overall population
        if self.n_comp_overall==3:
            ax = fig.add_subplot(8,1,1, projection='3d')
            
            for i, val in enumerate(num_labels_idx):
                idx_label = np.nonzero(y_vcf_idx==val)[0]
                pop_arr_idx = (np.where(self.pop_arr[:,1]==val)[0]).item()
                granular_pop = self.pop_arr[pop_arr_idx,2]

                ax.scatter(y_pred_overall[idx_label,0], y_pred_overall[idx_label,1], y_pred_overall[idx_label,2], s=55\
                        ,color=colors_pop[j], label = self.rev_pop_dict[granular_pop] )
                ax.scatter(y_target[idx_label,0], y_target[idx_label,1], y_target[idx_label,2], s=55\
                        ,color=colors_pop[j], marker='X')    
                j +=1

                lgnd = ax.legend(bbox_to_anchor=(0.9,0.5+(i/20)))
                for l in lgnd.legendHandles:
                    l._sizes = [30]

                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                # make the grid lines transparent
                ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

            ax.scatter(y_pred_overall[gradient_cp_idx,0], y_pred_overall[gradient_cp_idx,1], y_pred_overall[gradient_cp_idx,2], s=100,\
                color='black', marker='v')
            ax.set_title("Overall PCA space Predictions")
        else:
            ax = fig.add_subplot(8,1,1)
            for i, val in enumerate(num_labels_idx):
                idx_label = np.nonzero(y_vcf_idx==val)[0]
                pop_arr_idx = (np.where(self.pop_arr[:,1]==val)[0]).item()
                granular_pop = self.pop_arr[pop_arr_idx,2]

                ax.scatter(y_pred_overall[idx_label,0], y_pred_overall[idx_label,1], s=55\
                        ,color=colors_pop[j], label = self.rev_pop_dict[granular_pop] )
                ax.scatter(y_target[idx_label,0], y_target[idx_label,1], s=55\
                        ,color=colors_pop[j], marker='X')    
                j +=1

                lgnd = ax.legend(bbox_to_anchor=(0.9,0.5+(i/20)))
                for l in lgnd.legendHandles:
                    l._sizes = [30]
            ax.set_title("Overall PCA space Predictions")
        if self.n_comp_subclass>=2:
            # subplots for granular populations
            j = 0
            pop_num = np.arange(self.pop_order)
            for n, pop_num_val in enumerate(pop_num):
                wins_sp = np.nonzero(y_pred_sp==pop_num_val)
                if len(wins_sp)>1: 
                    wins_sp=wins_sp[1]
                    if isinstance(pop_num_val, list):
                        pop_name= "_".join([str(self.pop_order[i]) for i in pop_num_val])
                    else:
                        pop_name = self.pop_order[pop_num_val]
                    ax = fig.add_subplot(8,1,j+2, projection='3d')
                    j+=1
                    ax.scatter(y_pred_subclass[wins_sp,2*n], y_pred_subclass[wins_sp,2*n+1], y_pred_overall[wins_sp,0], s=55\
                                    , color=[colors_pop_dict[x] for x in y_vcf_idx[wins_sp,0]])
                    ax.scatter(y_target[wins_sp,2*n+self.n_comp_overall], y_target[wins_sp,2*n+1+self.n_comp_overall], \
                                y_target[wins_sp,0], s=75, color=[colors_pop_dict[x] for x in y_vcf_idx[wins_sp,0]], marker='X')
                    grad_wins_sp = np.intersect1d(wins_sp, gradient_cp_idx)
                    ax.scatter(y_pred_subclass[grad_wins_sp,2*n], y_pred_subclass[grad_wins_sp, 2*n+1], \
                        y_pred_overall[grad_wins_sp, 0], s=100, color='black', marker='v')
                    patches=[]
                    for i, val in enumerate(np.unique(y_vcf_idx[wins_sp,0])):
                        idx_label = (np.nonzero(self.pop_arr[:,1]==val)[0]).item()
                        pop_arr_idx = self.pop_arr[idx_label, 2]
                        granular_label = self.rev_pop_dict[pop_arr_idx] 
                        patches.append(mpatches.Patch(color = colors_pop_dict[val], label = granular_label))
                    
                    ax.legend(handles=patches, bbox_to_anchor=(0. ,0.9 ,1.,0.3),loc=10,ncol=7)
                    ax.set_title(f"{pop_name} PCA space Predictions")
                else:
                    pass
        fig_geo=None
        if self.geography:
            # add an axis in front for a single sample with batch dim=1
            xcord, ycord, zcord=y_pred_overall[np.newaxis,:,0], y_pred_overall[np.newaxis,:,1], \
                y_pred_overall[np.newaxis,:,2]
            coordinates = convert_coordinates(xcord, ycord, zcord)
            lbl_coord = convert_coordinates(y_target[np.newaxis,:,0], y_target[np.newaxis,:,1], \
                y_target[np.newaxis,:,2])
            coordinates=coordinates[0,...] #remove the first batch dim for plotting
            lbl_coord=lbl_coord[0,...]
            fig_geo=plot_coordinates_map(y_vcf_idx, coordinates, lbl_coord, self.rev_pop_dict, self.pop_arr)

        plt.subplots_adjust(hspace=0.1)
        plt.close('all')
        return fig, fig_geo

def plotAlleleCounts2Pops(ac1: np.ndarray, ac2:np.ndarray):
    """
    Rough check for relation between two pops
    """
    plt.scatter(ac1, ac2)
    plt.show()

def plotFst(Fst):
    ...