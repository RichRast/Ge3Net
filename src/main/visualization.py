import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import numpy as np
import seaborn as sns

def plot_coordinates_map(label, data_coordinates, rev_pop_order):
    """
    plotly plot for coordinates on a map
    label: target label vector (int) for the specific sample
    data_coordinates: lat, long for the specific sample
    rev_pop_order: dict with keys as target label ints and values
    as granular population name
    """
    fig = go.Figure(go.Scattergeo())
    label_list=np.unique(label)
    j=0
    for i in label_list:
        idx = np.nonzero(label)[0]
        
        fig.add_trace(go.Scattergeo(lon=data_coordinates[idx,1], lat=data_coordinates[idx,0]\
                                        ,text = rev_pop_order[i], name = rev_pop_order[i]))
        fig.update_traces(marker_size = 5)
        j +=1

    fig.show()
    
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

def plot_changepoint_predictions(y_pred_index_np, y_pred_var, cp_pred_index_np, bocd_cp, cp_mask, y_vcf_idx, rev_pop_dict, granular_pop_map):
    """
    plot for sample wise comparison of predicted changepoints
    and true changepoints for various methods
    """
    fig, ax = plt.subplots(figsize=[18,25])
    ax1 = fig.add_subplot(7,1,1)
    ax2 = fig.add_subplot(7,1,2)
    ax3 = fig.add_subplot(7,1,3)
    ax4 = fig.add_subplot(7,1,4)
    ax5 = fig.add_subplot(7,1,5)
    ax6 = fig.add_subplot(7,1,6)
    ax7 = fig.add_subplot(7,1,7)
    ax1.plot(y_pred_index_np)
    ax1.text(0, np.max(y_pred_index_np), s=rev_pop_dict[granular_pop_map[int(y_vcf_idx[1,:].item())]])
    for i in true_cps:
        ax1.plot([i,i], [np.min(y_pred_index_np), np.max(y_pred_index_np)], 'r' )
        ax1.text(i, np.max(y_pred_index_np), s=rev_pop_dict[granular_pop_map[int(y_vcf_idx[i+1,:].item())]])
    ax2.plot(y_pred_index_np[:-1,:]-y_pred_index_np[1:,:])
    ax3.plot(y_pred_var.detach().cpu().numpy()[index,:,:])
    ax4.plot(cp_pred_index_np)
    ax5.plot(np.arange(T+1), bocp_rl[index,:])
    ax6.plot(bocd_cp[index,:])
    ax7.plot(~cp_mask[index,:,0].bool())
    ax1.set_title("n_vectors")
    ax2.set_title("Simple gradient(post_process)")
    ax3.set_title("Mc dropout")
    ax4.set_title("Neural Network predicted cp")
    ax5.set_title("BOCD (post process) run_length")
    ax6.set_title("BOCD (post process) cp")
    ax7.set_title("cp target (cp_mask)")
    
    # log image/fig to wandb
    
    # plt.show()
    return fig

def predictions_plot(y_vcf_idx, y_pred, granular_pop_dict, pop_arr, PCA_lbls_dict):
    
    fig=plt.figure(figsize=(10,12))
    ax= Axes3D(fig)
    ax = fig.gca(projection='3d')

    colors1=['#0000ff','#008000', '#ff6dff', '#a3fecb', '#ff830c','#fc6500','#800080', '#ccccfe', '#4c4cfe', '#feccfe', '#000033', '#00fff2']
    
    num_labels_idx = np.unique(y_vcf_idx)
    print(f'num_labels_idx:{num_labels_idx}')

    colors1 = sns.color_palette("rainbow", len(num_labels_idx))
    color_pop_dict = {k:v for k,v in zip(num_labels_idx, colors1)}
    j =0
    # granular_pop_dict = pickle.load(, en_pickle=True)
    # PCA_lbls_dict = load_path(, en_pickle=True)
    # pop_arr = load_path()
    rev_pop_dict = {v:k for k,v in granular_pop_dict.items()}
    
    gradient_cp_idx = np.unique(np.where(abs(y_pred[:-1,:]-y_pred[1:,:])>0.3)[0])
    print(gradient_cp_idx)

    
    for i, val in enumerate(num_labels_idx):
        idx_label = np.nonzero(y_vcf_idx==val)[0]
        pop_arr_idx = np.where(pop_arr[:,1]==val)[0][0]
       
        ax.scatter(y_pred[idx_label,0], y_pred[idx_label,1], y_pred[idx_label,2], s=55\
                  ,color=colors1[j], label = rev_pop_dict[pop_arr[pop_arr_idx,2]] )
        ax.scatter(PCA_lbls_dict[val][0], PCA_lbls_dict[val][1], PCA_lbls_dict[val][2], s=55\
                  ,color=colors1[j], marker='X')    
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
        #ax.view_init(azim=-90, elev=19)
    
    ax.scatter(y_pred[gradient_cp_idx,0], y_pred[gradient_cp_idx,1], y_pred[gradient_cp_idx,2], s=100,\
           color='black', marker='v')
    
    # plt.show()
    return ax, fig

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

class Plot_per_epoch_revised(object):
    def __init__(self, n_comp_overall, n_comp_subclass, pop_order, rev_pop_dict, pop_arr):
        self.n_comp_overall = n_comp_overall
        self.n_comp_subclass = n_comp_subclass
        self.pop_order = pop_order
        self.rev_pop_dict = rev_pop_dict
        self.pop_arr = pop_arr
        
    def plot_index(self, y_pred_overall, y_pred_subclass, y_pred_sp, y_target, y_vcf_idx):
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
        
        plt.subplots_adjust(hspace=0.1)
        
        return fig, ax

# def plot_subclass(pop_order, pop_arr, PCA_lbls_dict, n_comp_overall, n_comp_subclass, rev_pop_order, wandb):
#     pop_num = np.arange(len(pop_order))
#     for j, pop_num_val in enumerate(pop_num):
#         # plot all the classes for the same subclass for train
#         # randomly select 30 granular pops for the particular subpop
        
#         if isinstance(pop_num_val, list):
#             pop_specific_idx = np.where(np.isin(pop_arr[:,3], pop_num_val))[0]
#             tmp_pop_name = "_".join([str(pop_order[i]) for i in pop_num_val])
#         else:
#             pop_specific_idx = np.where(pop_arr[:,3]==pop_num_val)[0]
#             tmp_pop_name = pop_order[pop_num_val]                
#         random_idx = np.random.choice(pop_arr[pop_specific_idx,1], 30)
#         PCA_labels = np.array(list(PCA_lbls_dict.values()), dtype=float)
#         ax1, fig = plot_embeddings_2d_extended(PCA_labels[:, n_comp_overall+n_comp_subclass*j:n_comp_overall+n_comp_subclass*(j+1)], \
#             pop_arr, pop_order)
        
#         for k in random_idx:
#             idx_pop_arr=np.where(pop_arr[:,1]==k)[0][0]
#             ax1.text(PCA_lbls_dict[k][n_comp_overall+n_comp_subclass*j], PCA_lbls_dict[k][n_comp_overall+1+n_comp_subclass*j], \
#                     s = rev_pop_order[pop_arr[idx_pop_arr,2]],\
#                 fontweight='bold', fontsize = 12)

#         plt.title(f" subclass : {tmp_pop_name}")
#         plt.show()
#         if wandb is not None:
#             fig_image_subclass = wandb.Image(fig)
#             wandb.log({f"subclass for {tmp_pop_name}":fig_image_subclass})