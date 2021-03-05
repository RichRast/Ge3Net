import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np

def plot_embeddings(X_transformed, pop_arr, n_way):
    fig=plt.figure(figsize=(10,12))
    plt.rcParams['savefig.transparent'] = True
    ax= Axes3D(fig)
    ax = fig.gca(projection='3d')
    
    colors_pop=['#0000ff','#008000', '#ff6dff', '#a3fecb', '#998fff','#fc6500','#800080']
    pop = ['EAS', 'SAS', 'WAS', 'OCE', 'AFR', 'AMR', 'EUR']
    color_pop_dict = {k:v for k,v in zip(pop, colors_pop)}

    for i in range(n_way):
        idx_label = np.nonzero(pop_arr==i)[0]
        ax.scatter(X_transformed[idx_label,0], X_transformed[idx_label,1],X_transformed[idx_label,2], s=5,\
                  color=color_pop_dict[pop[i]] , label = pop[i])
        
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

    return ax

def plot_embeddings_2d(X_transformed, pop_arr, n_way):
    plt.rcParams['savefig.transparent'] = True
    fig, ax= plt.subplots(figsize=(10,12))
    
    colors_pop=['#0000ff','#008000', '#ff6dff', '#a3fecb', '#998fff','#fc6500','#800080']
    pop = ['EAS', 'SAS', 'WAS', 'OCE', 'AFR', 'AMR', 'EUR']
    color_pop_dict = {k:v for k,v in zip(pop, colors_pop)}

    for i in range(n_way):
        idx_label = np.nonzero(pop_arr==i)[0]
        ax.scatter(X_transformed[idx_label,0], X_transformed[idx_label,1], s=5,\
                  color=color_pop_dict[pop[i]] , label = pop[i])
        
        lgnd = ax.legend(bbox_to_anchor=(0.9,0.5+(i/20)))
        for l in lgnd.legendHandles:
            l._sizes = [30]

        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # # make the grid lines transparent
        # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    return ax

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
    
def plot_dist(mean, std , chm):
    """
    plots the distribution (mean and std)
    for a given chm
    """
    fig, (ax1, ax2) = plt.subplots(2)
    plt.rcParams['figure.figsize']=[18,15]
    count, bins, ignored = ax1.hist(mean, 20, density = True)
    ax1.set_title(f"Mean distribution for chm {chm}")
    count, bins, ignored = ax2.hist(std, 10, density = True)
    ax2.set_title(f"Std dev. distribution for chm {chm}")
    plt.show()

def plot changepoint_predictions(y_pred_index_np, y_pred_var, cp_pred_index_np, bocd_cp, cp_mask, y_vcf_idx, rev_pop_dict, granular_pop_map):
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
    plt.show()