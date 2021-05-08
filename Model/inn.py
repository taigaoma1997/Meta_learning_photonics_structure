from utils import count_params, weights_init_normal
from utils import MMD_multiscale
from training import Trainer, GANTrainer, INNTrainer
from datasets import SiliconColor, get_dataloaders
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
from torch.optim.lr_scheduler import StepLR
from models import MLP, TandemNet, cVAE, cGAN, INN, cVAE_new, cVAE_GSNN, cVAE_GSNN1, cVAE_Full, cVAE_tandem, cVAE_hybrid
from utils import evaluate_simple_inverse, evaluate_tandem_accuracy, evaluate_vae_inverse, evaluate_gan_inverse, evaluate_inn_inverse
from utils import evaluate_forward_minmax_dataset, evaluate_gan_minmax_inverse, evaluate_inn_minmax_inverse, evaluate_tandem_minmax_accuracy, evaluate_vae_GSNN_minmax_inverse, evaluate_forward_minmax

from configs import get_configs
import random

import sys
import torch
import os
from torch import nn
import numpy as np

import argparse

torch.manual_seed(random.randint(1,1000))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# For INN model

# To speficy which GPU to use, run with: CUDA_VISIBLE_DEVICES=5,6 python inn.py

def train(model, train_loader, optimizer, criterion, current_epoch, configs):

    epochs = configs.epochs

    lambd_predict = configs.lambd_predict
    lambd_latent = configs.lambd_latent
    lambd_rev = configs.lambd_rev

    latent_criterion = MMD_multiscale
    backward_criterion = MMD_multiscale

    model.train()

    loss_epoch = 0
    loss_factor = min(1., 2. * 0.002 ** (1. - (float(current_epoch) / epochs)))

    for x, y in train_loader:
        batch_size = len(x)

        optimizer.zero_grad()
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_clean = y.clone()

        pad_x, pad_yz = model.create_padding(batch_size)
        pad_x, pad_yz = pad_x.to(DEVICE), pad_yz.to(DEVICE)

        y += model.y_noise_scale * torch.randn(batch_size, y.size(1)).float().to(DEVICE)

        x = torch.cat((x, pad_x), dim = 1)
        y = torch.cat((torch.randn(batch_size, model.dim_z).float().to(DEVICE), pad_yz, y), dim = 1)

        optimizer.zero_grad()

        output = model(x)[0]

        y_short = torch.cat((y[:, :model.dim_z], y[:, -model.dim_y:]), dim = 1)

        l = lambd_predict * criterion(output[:, model.dim_z:], y[:, model.dim_z:])
        output_block_grad = torch.cat((output[:, :model.dim_z],
                                           output[:, -model.dim_y:].data), dim=1)

        l += lambd_latent * latent_criterion(output_block_grad, y_short)
        loss_epoch += l.data.item() * len(x)

        l.backward()

        # Backward step:
        pad_yz = model.zeros_noise_scale * \
            torch.randn(batch_size, model.ndim_total -
                        model.dim_y - model.dim_z).to(DEVICE)
        y = y_clean + model.y_noise_scale * \
            torch.randn(batch_size, model.dim_y).to(DEVICE)

        orig_z_perturbed = (output.data[:, :model.dim_z] + model.y_noise_scale * torch.randn(batch_size, model.dim_z).to(DEVICE))
        y_rev = torch.cat((orig_z_perturbed, pad_yz,y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, model.dim_z).to(DEVICE), pad_yz, y), dim=1)
            
        output_rev = model(y_rev, rev=True)[0]
        output_rev_rand = model(y_rev_rand, rev=True)[0]

        l_rev = (lambd_rev * loss_factor * backward_criterion(output_rev_rand[:, :model.dim_x],x[:, :model.dim_x]))

        l_rev += lambd_predict * criterion(output_rev, x)
            
        loss_epoch += l_rev.data.item() * len(x)
        l_rev.backward()

        for p in model.parameters():
            if p.grad!=None:               
                p.grad.data.clamp_(-15.00, 15.00)

        optimizer.step()


    return loss_epoch / len(train_loader.dataset)


def evaluate(model, val_loader, test_loader, optimizer, criterion, forward_model, configs, test=False):
    
    epochs = configs.epochs

    lambd_predict = configs.lambd_predict
    lambd_latent = configs.lambd_latent
    lambd_rev = configs.lambd_rev

    latent_criterion = MMD_multiscale
    backward_criterion = MMD_multiscale

    model.eval()
    dataloader = test_loader if test else val_loader

    loss_epoch = 0
    loss_factor = 1
    rmse_cie_raw = 0
    with torch.no_grad():
        for x, y in dataloader:

            batch_size = len(x)

            x, y = x.to(DEVICE), y.to(DEVICE)
            y_clean = y.clone()

            pad_x, pad_yz = model.create_padding(batch_size)
            pad_x, pad_yz = pad_x.to(DEVICE), pad_yz.to(DEVICE)

            y += model.y_noise_scale * torch.randn(batch_size, y.size(1)).float().to(DEVICE)

            x = torch.cat((x, pad_x), dim = 1)
            y = torch.cat((torch.randn(batch_size, model.dim_z).float().to(DEVICE), pad_yz, y), dim = 1)

            output = model(x)[0]

            y_short = torch.cat((y[:, :model.dim_z], y[:, -model.dim_y:]), dim = 1)

            l = lambd_predict * criterion(output[:, model.dim_z:], y[:, model.dim_z:])

            output_block_grad = torch.cat((output[:, :model.dim_z],
                                        output[:, -model.dim_y:].data), dim=1)

            l += lambd_latent * latent_criterion(output_block_grad, y_short)
            loss_epoch += l.data.item() * len(x)
            
            # Backward step:
            pad_yz = model.zeros_noise_scale * \
                torch.randn(batch_size,  model.ndim_total -
                             model.dim_y -  model.dim_z).to(DEVICE)
            y = y_clean +  model.y_noise_scale * \
                torch.randn(batch_size,  model.dim_y).to(DEVICE)

            orig_z_perturbed = (output.data[:, : model.dim_z] +  model.y_noise_scale * torch.randn(
                batch_size,  model.dim_z).to(DEVICE))
            y_rev = torch.cat((orig_z_perturbed, pad_yz, y), dim=1)
            y_rev_rand = torch.cat((torch.randn(batch_size,  model.dim_z).to(DEVICE), pad_yz, y), dim=1)
            
            output_rev =  model(y_rev, rev=True)[0]
            output_rev_rand =  model(y_rev_rand, rev=True)[0]

            l_rev = (
                 lambd_rev
                * loss_factor
                *  backward_criterion(output_rev_rand[:, : model.dim_x],
                                x[:, : model.dim_x])
            )

            l_rev +=  lambd_predict *  criterion(output_rev, x)
            
            loss_epoch += l_rev.data.item() * len(x)

            for p in  model.parameters():
                if p.grad!=None:
                    p.grad.data.clamp_(-15.00, 15.00)
            
        #cie_raw, param_raw, cie_pred, param_pred = evaluate_inn_minmax_inverse(model, forward_model, val_loader.dataset, show=0)
        
        #rmse_cie_raw = np.sqrt(np.sum(np.average(np.square((cie_raw - cie_pred)),axis=0)))

    return loss_epoch / len(dataloader.dataset), rmse_cie_raw


def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_all':loss_all,
            'configs':configs,
        }, path)


def main(configs):
    
    train_loader, val_loader, test_loader = get_dataloaders('inn', configs.batch_size)

    model = INN(configs.ndim_total, configs.input_dim, configs.output_dim, dim_z = configs.latent_dim).to(DEVICE)
    
    # set up optimizer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    
    if configs.if_lr_de:
        scheduler = StepLR(optimizer, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        
    criterion = torch.nn.MSELoss()


    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
    forward_model = MLP(4, 3).to(DEVICE)
    forward_model.load_state_dict(torch.load('./models/forward_model_trained_evaluate_3.pth')['model_state_dict'])

    # start training 
    path =  './models/inn/inn_latent_'+str(configs.latent_dim)+'_lr_'+str(configs.lr)+'_STEP_'+ str(configs.if_lr_de) +'_trained.pth'
    path_temp = './models/inn/inn_latent_'+str(configs.latent_dim)+'_lr_'+str(configs.lr)+'_STEP_'+ str(configs.if_lr_de) +'_trained_temp.pth'
    epochs = configs.epochs
    loss_all = np.zeros([3, configs.epochs])
    loss_val_best = 100
    
    for e in range(epochs):
        
        loss_train = train(model, train_loader, optimizer, criterion, e, configs)
        loss_val, loss_val_2 = evaluate(model, val_loader, test_loader, optimizer, criterion, forward_model, configs)
        loss_all[0,e] = loss_train
        loss_all[1,e] = loss_val
        loss_all[2,e] = loss_val_2

        if loss_val_best >= loss_all[1,e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[1,e]
            save_checkpoint(model, optimizer, e, loss_all, path, configs)

        print('Epoch {}, train loss {:.6f}, val loss {:.6f}.'.format(e, loss_train, loss_val))

        if e%10==0:
            save_checkpoint(model, optimizer, e, loss_all, path_temp, configs)

        if configs.if_lr_de:
            scheduler.step()




if __name__  == '__main__':

    parser = argparse.ArgumentParser('nn models for inverse design: Invertible neural network')
    parser.add_argument('--model', type=str, default='INN')
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension of condition for the generator')
    parser.add_argument('--output_dim', type=int, default=3, help='Output size of generator')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of dataset')
    parser.add_argument('--ndim_total', type=int, default=16, help='??')
    parser.add_argument('--latent_dim', type=int, default=2, help='???')
    parser.add_argument('--lambd_predict', type=int, default=3, help='??')
    parser.add_argument('--lambd_latent', type=int, default=400, help='??')
    parser.add_argument('--lambd_rev', type=int, default=400, help='??')

    parser.add_argument('--epochs', type=int, default=10000, help='Number of iteration steps')

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--if_lr_de',action='store_true', default='False', help='If decrease learning rate duing training')
    parser.add_argument('--lr_de', type=float, default=0.2, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=5000, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    args = parser.parse_args()
    

    main(args)
