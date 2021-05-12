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

def plot_cie_raw_pred_1(cie_raw, cie_pred):
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
    
def plot_cie_raw_pred(cie_raw, cie_pred):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    titles = ['x', 'y', 'Y']
    xlim = [[0.1, 0.6],[0.0,0.8],[0.0,0.7]]
    for i in range(3):
        raw_pred = np.array(sorted(zip(cie_raw[:, i], cie_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1], s =3 )
        ax[i].plot(xlim[i],xlim[i], c='k')
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
        ax[i].set_xlim(xlim[i])
        ax[i].set_ylim(xlim[i])
    plt.show()
    

def plot_struc_raw_pred(param_raw, param_pred, a = 1):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    titles = ['Height', 'Gap', 'Period','Diamater']
    xlim = [[0, 210], [150, 330], [280, 720], [75, 165]]
    for i in range(4):
        raw_pred = np.array(sorted(zip(param_raw[:, i], param_pred[:, i])))
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
    
def plt_abs_err(CIE_x, cie_pred):
    abs_err = abs(CIE_x - cie_pred)
    abs_mean = sum(abs_err)/len(abs_err)
    
    plt.figure(figsize = [8, 7])
    plt.subplot(3,1 ,1)
    plt.scatter(CIE_x[:,0],abs_err[:,0], color='r',label='x')
    plt.axhline(y=abs_mean[0],color='r', linestyle='-')
    plt.text(0.5,abs_mean[0] , str(round(abs_mean[0],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.scatter(CIE_x[:,1],abs_err[:,1], color='g',label='y')
    plt.axhline(y=abs_mean[1], color='g',linestyle='-')
    plt.text(0.6,abs_mean[1] , str(round(abs_mean[1],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1, 3)
    plt.scatter(CIE_x[:,2],abs_err[:,2], color='b',label='Y')
    plt.axhline(y=abs_mean[2], color='b',linestyle='-')
    plt.text(0.6,abs_mean[2] , str(round(abs_mean[2],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

def plt_abs_percent_err(CIE_x, cie_pred):
    abs_err = abs(CIE_x - cie_pred)/CIE_x
    abs_mean = sum(abs_err)/len(abs_err)
    
    plt.figure(figsize = [8, 7])
    plt.subplot(3,1 ,1)
    plt.scatter(CIE_x[:,0],abs_err[:,0], color='r',label='x')
    plt.axhline(y=abs_mean[0],color='r', linestyle='-')
    plt.text(0.5,abs_mean[0] , str(round(abs_mean[0],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.scatter(CIE_x[:,1],abs_err[:,1], color='g',label='y')
    plt.axhline(y=abs_mean[1], color='g',linestyle='-')
    plt.text(0.6,abs_mean[1] , str(round(abs_mean[1],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1, 3)
    plt.scatter(CIE_x[:,2],abs_err[:,2], color='b',label='Y')
    plt.axhline(y=abs_mean[2], color='b',linestyle='-')
    plt.text(0.6,abs_mean[2] , str(round(abs_mean[2],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

    
def plt_hist_struc(param, labels):
    plt.figure(figsize = [20, 3])
    plt.subplot(1, 4,1)
    plt.hist(param[:,0], bins=20, histtype='step', label=labels)
    plt.title('Height histogram')
    plt.xlabel('Height/(nm)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 4,2)
    plt.hist(param[:,1], bins=20, histtype='step', label=labels)
    plt.title('Gap histogram')
    plt.xlabel('Gap/(nm)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.hist(param[:,2], bins=20, histtype='step', label=labels)
    plt.title('Period histogram')
    plt.xlabel('Period/(nm)')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.hist(param[:,3], bins=20, histtype='step', label=labels)
    plt.title('Diameter histogram')
    plt.xlabel('Diamater/(nm)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    