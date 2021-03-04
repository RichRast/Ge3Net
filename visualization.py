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

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)

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