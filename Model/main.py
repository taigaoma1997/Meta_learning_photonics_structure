from utils import count_params, weights_init_normal
from training import Trainer, GANTrainer, INNTrainer
from datasets import SiliconColor, get_dataloaders
from models import MLP, TandemNet, cVAE, cGAN, INN, cVAE_new, cVAE_GSNN, cVAE_Full, cVAE_tandem, cVAE_hybrid
from configs import get_configs
import random

import sys
import torch
from torch import nn

import argparse

torch.manual_seed(random.randint(1,100))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser('nn models for inverse design')
    parser.add_argument('--model', type=str, default='forward_model')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(args.model)
    configs = get_configs(args.model)

    if args.model in ['forward_model', 'inverse_model']:
        model = MLP(configs['input_dim'], configs['output_dim']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])
    
    elif args.model in ['tandem_net']:

        forward_model = MLP(4, 3).to(DEVICE)
        forward_model.load_state_dict(torch.load('./models/forward_model_trained.pth')['model_state_dict'])
        inverse_model = MLP(3, 4).to(DEVICE)
        #inverse_model.load_state_dict(torch.load('./models/inverse_model_trained.pth')['model_state_dict'])
        model = TandemNet(forward_model, inverse_model)
        optimizer = torch.optim.Adam(model.inverse_model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])
        
    
    elif args.model in ['vae']:
        model = cVAE(configs['input_dim'], configs['latent_dim']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])
        
    elif args.model in ['vae_new']:
        model = cVAE_new(configs['input_dim'], configs['latent_dim']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])
        
    elif args.model in ['vae_GSNN']:
        model = cVAE_GSNN(configs['input_dim'], configs['latent_dim']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])

    elif args.model in ['vae_Full']:
        model = cVAE_Full(configs['input_dim'], configs['latent_dim']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])

    elif args.model in ['vae_tandem']:
        model = cVAE_tandem(configs['input_dim'], configs['latent_dim']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])

    elif args.model in ['vae_hybrid']:
        forward_model = MLP(4, 3).to(DEVICE)
        forward_model.load_state_dict(torch.load('./models/forward_model_trained.pth')['model_state_dict'])
        vae_model = cVAE_GSNN(configs['input_dim'], configs['latent_dim']).to(DEVICE)
        #vae_model.load_state_dict(torch.load('./models/vae_GSNN_trained_4.pth')['model_state_dict'])
        model = cVAE_hybrid(forward_model, vae_model)
        optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])






    elif args.model in ['gan']:
        model = cGAN(configs['input_dim'], configs['output_dim'], configs['noise_dim']).to(DEVICE)
        model.apply(weights_init_normal)

        optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=configs['g_learning_rate'], weight_decay=configs['weight_decay'])
        optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=configs['d_learning_rate'], weight_decay=configs['weight_decay'])

        print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
        criterion = torch.nn.BCELoss()
        trainer = GANTrainer(model, optimizer_G, optimizer_D, train_loader, val_loader, test_loader, criterion, configs['epochs'], args.model)
        trainer.fit()
        sys.exit(0)

    elif args.model in ['inn']:
        
        model = INN(configs['ndim_total'], configs['input_dim'], configs['output_dim'], dim_z = configs['latent_dim']).to(DEVICE)
        print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])

        criterion = torch.nn.MSELoss()
        trainer = INNTrainer(model, optimizer, train_loader, val_loader, test_loader, criterion, configs['epochs'], args.model)
        #trainer.fit()
        trainer.fit_inn()
        sys.exit(0)

    else:
        raise NameError
    
    print('Model {}, Number of parameters {}'.format(args.model, count_params(model)))
    criterion = nn.MSELoss()
    trainer = Trainer(model, optimizer, train_loader, val_loader, test_loader, criterion, configs['epochs'], args.model)
    # train the model 
    trainer.fit()


    