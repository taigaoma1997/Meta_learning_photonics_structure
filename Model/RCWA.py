# Save loaded data. Don't run this!!!
import numpy as np
#import matplotlib.pyplot as pyplot
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from configs import get_configs
from models import MLP, TandemNet, cVAE, cGAN, INN, cVAE_new, cVAE_GSNN, cVAE_Full, cVAE_hybrid, cVAE_tandem
from utils import evaluate_simple_inverse, evaluate_tandem_accuracy, evaluate_vae_inverse,evaluate_vae_GSNN_minmax_inverse, evaluate_gan_inverse, evaluate_inn_inverse
from datasets import get_dataloaders, SiliconColor
import scipy.io     # used to load .mat data
from scipy.io import savemat
train_loader, val_loader, test_loader = get_dataloaders('forward_model')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

forward_model = MLP(4, 3).to(DEVICE)
forward_model.load_state_dict(torch.load('./models/forward_model_trained_evaluate_3.pth', map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))['model_state_dict'])

def struc_check(structure):
    if np.sum(abs(structure)-structure)>0:  # if get negative parameters, then wrong structure
        return 0
    else:
        struc = np.reshape(structure, (-1, 4));
        N = np.shape(struc)[0]
        #print(struc)
        for i in range(N):
            if (struc[i,1]+struc[i,3]>=struc[i,2]):  # if gap+diameter >= period, then wrong structure
                return 0;
            
        return 1;
    
def struc_remove(cie_raw, param_raw, param_pred):
    # remove all structures predicted that is not satisfied by struc_check
    M = np.shape(cie_raw)[0]
    j = 0
    B = []
    for i in range(M-1):
        if struc_check(param_pred[i,:])==0:
            param_pred[j,:] = param_pred[i+1,:]
            param_raw[j,:] = param_raw[i+1,:]
            cie_raw[j,:] = cie_raw[i+1,:]
            B.append(i)
            print(i,j)
        else:
            param_pred[j,:] = param_pred[i,:]
            param_raw[j,:] = param_raw[i,:]
            cie_raw[j,:] = cie_raw[i,:]
            j = j+1
    i = i+1
    
    if struc_check(param_pred[i,:])==0:
        param_pred = param_pred[0:j,:]
        param_raw = param_raw[0:j,:]
        cie_raw = cie_raw[0:j,:]
    else:
        param_pred[j,:] = param_pred[i,:]
        param_raw[j,:] = param_raw[i,:]
        cie_raw[j,:] = cie_raw[i,:]
        param_pred = param_pred[0:(j+1),:]
        param_raw = param_raw[0:(j+1),:]
        cie_raw = cie_raw[0:(j+1),:]

    return B, cie_raw, param_raw, param_pred
print('Okay')



M = 1411

total = 10
param_pred_all = np.zeros([M, 4*total])
cie_pred_all = np.zeros([M, 3*total])

configs = get_configs('vae_GSNN')
vae_model = cVAE_GSNN(configs['input_dim'], configs['latent_dim']).to(DEVICE)
vae_model.load_state_dict(torch.load('./models/' + 'vae_GSNN' + '_trained_4.pth')['model_state_dict'])

for i in range(total):
    cie_raw, param_raw, cie_pred, param_pred =  evaluate_vae_GSNN_minmax_inverse(vae_model, forward_model,test_loader.dataset)
    cie_pred_all[:,(3*i):(3*i+3)], param_pred_all[:,(4*i):(4*i+4)]= cie_pred, param_pred




def RCWA(cie_raw, param_raw, cie_pred, param_pred, cie_pred_all, param_pred_all, model):
    
    mdic = {"param_pred_all": param_pred_all,"cie_pre_all":cie_pred_all,"param_test_all": param_raw,"CIE_x_all": cie_raw}
    savemat("data_predicted/param_" + model + "_pred_all.mat",mdic)


    B, temp_1, temp_2, temp_3= struc_remove(cie_raw, param_raw, param_pred_all)
    cie_pred_all = np.delete(cie_pred_all, B, 0)
    cie_raw = np.delete(cie_raw, B, 0)
    param_pred_all = np.delete(param_pred_all, B, 0)
    param_raw = np.delete(param_raw, B, 0)

# Saving the predicted data
    mdic = {"param_pred": param_pred_all,"param_test": param_raw,"cie_pred": cie_pred_all, "CIE_x": cie_raw, "deleted_row":B}
    savemat("data_predicted/param_" + model + "_pred.mat",mdic)

    print(param_pred[1,:])

    np.shape(param_pred)

    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('./RCWA'))
    eng.MODEL_vae(model)
    eng.quit()

    filepath ="data_predicted/xyY/xyY_param_" + model + "_pred.mat"
    temp = scipy.io.loadmat(filepath)
    return temp

RCWA(cie_raw, param_raw, cie_pred, param_pred, cie_pred_all, param_pred_all,'vae_GSNN')