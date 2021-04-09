import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colour

from sklearn.metrics import r2_score


def compare_param_dist(param_raw, param_pred):
    len_param = len(param_raw)
    category = (['height'] * len_param + ['gap'] * len_param + ['period'] * len_param + ['diameter'] * len_param) * 2
    param_raw = param_raw.T.reshape(-1)
    param_pred = param_pred.T.reshape(-1)
    type = ['raw'] * len(param_pred) + ['pred'] * len(param_pred)
    res_df = pd.DataFrame({'val':np.concatenate((param_raw, param_pred)), 'cat':category, 'type':type})
    sns.boxplot(x='cat', y='val', data=res_df, hue='type')

def compare_cie_dist(cie_raw, cie_pred):
    len_cie = len(cie_raw)
    category = (['x'] * len_cie + ['y'] * len_cie + ['Y'] * len_cie) * 2
    cie_raw = cie_raw.T.reshape(-1)
    cie_pred = cie_pred.T.reshape(-1)
    type = ['raw'] * len(cie_pred) + ['pred'] * len(cie_pred)
    res_df = pd.DataFrame({'val':np.concatenate((cie_raw, cie_pred)), 'cat':category, 'type':type})
    sns.boxplot(x='cat', y='val', data=res_df, hue='type')

def plot_cie(cie_raw, cie_pred):
    from colour.plotting import plot_chromaticity_diagram_CIE1931
    from matplotlib.patches import Polygon

    fig, ax = plot_chromaticity_diagram_CIE1931()
    srgb = Polygon(list(zip([0.64, 0.3, 0.15], [0.33, 0.6, 0.06])), facecolor='0.9', alpha=0.1, edgecolor='k')
    ax.add_patch(srgb)
    ax.scatter(cie_raw[:,0], cie_raw[:,1], s=1, c='b')
    ax.scatter(cie_pred[:,0], cie_pred[:,1], s=1, c='k')
    return fig

def plot_cie_raw_pred(cie_raw, cie_pred):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    titles = ['x', 'y', 'Y']
    for i in range(3):
        raw_pred = np.array(sorted(zip(cie_raw[:, i], cie_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1])
        ax[i].plot([raw_pred[:,0].min(), raw_pred[:,0].max()], [raw_pred[:,1].min(), raw_pred[:,1].max()], c='k')
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
    plt.show()
    

def plot_struc_raw_pred(cie_raw, cie_pred, a = 1):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    titles = ['Height', 'Gap', 'Period','Diamater']
    xlim = [[0, 210], [150, 330], [280, 720], [75, 165]]
    for i in range(4):
        raw_pred = np.array(sorted(zip(cie_raw[:, i], cie_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1])
        
        
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
        if a==1:
            ax[i].set_xlim(xlim[i])
            ax[i].set_ylim(xlim[i])
            ax[i].plot(xlim[i],xlim[i], c='k')
            ax[i].set_title(titles[i])
            ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
            continue
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
    plt.show()