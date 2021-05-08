import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
from utils import MMD_multiscale
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from configs import get_configs

from models import MLP, TandemNet, cVAE, cGAN, INN, cVAE_GSNN, cVAE_hybrid
from utils import evaluate_simple_inverse, evaluate_tandem_accuracy, evaluate_vae_inverse, evaluate_gan_inverse, evaluate_inn_inverse
from utils import evaluate_forward_minmax_dataset, evaluate_gan_minmax_inverse, evaluate_tandem_minmax_accuracy, evaluate_vae_GSNN_minmax_inverse, evaluate_forward_minmax

from configs import get_configs
from plotting_utils import compare_cie_dist, compare_param_dist, plot_cie, plot_cie_raw_pred, plt_abs_err
from datasets import get_dataloaders, SiliconColor



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# trainer for forward, direct inverse, tandem, and vae models
class Trainer():

    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 val_loader,
                 test_loader,
                 criterion,
                 epochs,
                 model_name):

        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.epochs = epochs
        self.path = './models/' + model_name + '_trained.pth'
        self.temp_path = './models/' + model_name + '_trained_temp.pth'

    def train(self):
        # x: structure ; y: CIE 
        if self.model_name == 'tandem_net':
            self.model.inverse_model.train()
            self.model.forward_model.eval()
        elif self.model_name == 'vae_hybrid':
            self.model.vae_model.train()
            self.model.forward_model.eval()
        else:
            self.model.train()
            
        loss_epoch = 0
        for x, y in self.train_loader:
            
            if self.current_epoch%1000 == 0:
                times = int(self.current_epoch/1000)
                if self.model_name =='forward_model':
                    configs = get_configs('forward_model')
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=pow(0.2, times)*configs['learning_rate'], weight_decay=configs['weight_decay'])
                elif self.model_name =='tandem_net':
                    configs = get_configs('forward_model')
                    self.optimizer = torch.optim.Adam(self.model.inverse_model.parameters(), lr=pow(0.2, times)*configs['learning_rate'], weight_decay=configs['weight_decay'])
                elif self.model_name =='vae_hybrid':
                    configs = get_configs('vae_hybrid')
                    self.optimizer = torch.optim.Adam(self.model.vae_model.parameters(), lr=pow(0.2, times)*configs['learning_rate'], weight_decay=configs['weight_decay'])
            
            self.optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = self.model(x, y)
            loss = self.get_loss(x, y, pred)
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.to('cpu').item() * len(x)

        return loss_epoch / len(self.train_loader.dataset) 
    
    
    def evaluate(self, test = False):
         # x: structure ; y: CIE 
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        loss_epoch = 0
        with torch.no_grad():
            for x, y in dataloader:

                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = self.model(x,y)
                loss = self.get_loss(x, y, pred)
                loss_epoch += loss.to('cpu').item() * len(x)

        return loss_epoch / len(dataloader.dataset)
    

    def fit_tandem(self):
        temp1 = np.zeros([2,self.epochs]);
        for e in range(self.epochs):
            
            forward_model1 = self.model.forward_model
            #cie_pred, cie_raw = self.evaluate_minmax_forward_dataset(self.model.forward_model, self.test_loader.dataset)
            
            dataset = self.test_loader.dataset
            with torch.no_grad():
                range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
                x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
                x_dim = x.size()[1]
                M = x.size()[0]
                print('-----------------------------')
                x_pred = forward_model1.forward(y, None)
                
                x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
                x_raw = x *range_[:x_dim] + min_[:x_dim]
                
            x_pred_raw = x_pred_raw.cpu().numpy()
            x_raw = x_raw.cpu().numpy()
            
            cie_raw = x_raw
            cie_pred = x_pred_raw
            print('raw', cie_raw, '\n pred',cie_pred)
            print(forward_model1.state_dict())
            plot_cie_raw_pred(cie_raw, cie_pred)
            
            
            
            loss_train = self.train_tandem()
            
            forward_model2 = self.model.forward_model
            cie_pred, cie_raw = evaluate_minmax_forward_dataset(self.model.forward_model, self.test_loader.dataset)
            
            
            self.difference_param(forward_model1, forward_model2)
            #plot_cie_raw_pred(cie_raw, cie_pred)
            #print(list(self.model.forward_model.parameters()))
            loss_val = self.evaluate()

            temp1[0,e] = loss_train
            temp1[1,e] = loss_val
            
            print('Epoch {}, train loss {:.5f}, val loss {:.5f}'.format(
                e, loss_train, loss_val))
        
        plt.plot(range(self.epochs),temp1[0,:],label='Training loss')  
        plt.plot(range(self.epochs),temp1[1,:],label='Val Loss')                      
        # plot the training and val loss VS epoches.
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        
        loss_test = self.evaluate(test=True)
        print('Training finished! Test loss {:.3f}'.format(loss_test))

        self.save_checkpoint(e, loss_test)
        print('Saved final trained model.')

    def get_loss(self, x, y, pred):
        '''
        Loss for training simple forward and inverse networks.
        '''
        
        if self.model_name in ['inverse_model', 'forward_model', 'tandem_net']:
            return self.criterion(pred, y)
        
        elif self.model_name == 'vae':
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            recon_x, mu, logvar, y_pred = pred
            recon_loss = self.criterion(recon_x, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = self.criterion(y_pred, y)
            return recon_loss + KLD + pred_loss
        
        elif self.model_name == 'vae_new':
            recon_x, mu, logvar, y_pred = pred
            recon_loss = self.criterion(recon_x, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + KLD
        
        elif self.model_name == 'vae_GSNN':
            recon_x, mu, logvar, x_pred = pred
            recon_loss = self.criterion(recon_x, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = self.criterion(x_pred, x)
            return recon_loss + KLD+ pred_loss
        
        elif self.model_name == 'vae_Full':
            recon_x, mu, logvar, y_pred = pred
            recon_loss = self.criterion(recon_x, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + KLD
        
        elif self.model_name =='vae_tandem':
            recon_x, mu, logvar, x_pred, y_pred = pred
            recon_loss = self.criterion(recon_x, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = self.criterion(x_pred, x)+ self.criterion(y_pred, y)
            return recon_loss + KLD+ pred_loss
        
        elif self.model_name =='vae_hybrid':
            recon_x, mu, logvar, x_hat,  y_pred = pred
            recon_loss = self.criterion(recon_x, x)
            replace_loss = self.criterion(x_hat, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = self.criterion(y_pred, y)
            
            return recon_loss + replace_loss + KLD + 5*pred_loss        
        
        else:
            raise NotImplementedError
            
            
    def fit(self):
        self.loss_all = np.zeros([2,self.epochs])
        loss_val_best = 100

        for e in range(self.epochs):
            
            self.current_epoch = e
            loss_train = self.train()
            loss_val = self.evaluate()
            self.loss_all[0,e] = loss_train
            self.loss_all[1,e] = loss_val

            if loss_val_best >= self.loss_all[1,e]:
            # save the best model for smallest validation loss
                loss_val_best = self.loss_all[1,e]
                loss_test = self.evaluate(test=True)
                self.save_checkpoint(e, loss_test, self.loss_all, self.path)

            print('Epoch {}, train loss {:.6f}, val loss {:.6f}'.format(e, loss_train, loss_val))
                

            if e%10==0:
                self.save_checkpoint(e, loss_val_best, self.loss_all, self.temp_path)
        
        plt.plot(range(self.epochs),self.loss_all[0,:],label='Training loss')  
        plt.plot(range(self.epochs),self.loss_all[1,:],label='Val Loss')                      
        # plot the training and val loss VS epoches.
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        self.plot_result()
        
        
    def plot_result(self):
        # plot the evaluation of trained model on test data
        if self.model_name in ['forward_model']:
            forward_model = MLP(4, 3).to(DEVICE)
            forward_model.load_state_dict(torch.load(self.path)['model_state_dict'])
            cie_pred, cie_raw = evaluate_forward_minmax_dataset(forward_model, self.test_loader.dataset)
            plot_cie_raw_pred(cie_raw, cie_pred) # compare the r2 sore
            plt_abs_err(cie_raw, cie_pred)  # compare the absolute mean 
        elif self.model_name in ['tandem_net']:
            forward_model = MLP(4, 3).to(DEVICE)
            inverse_model = MLP(3, 4).to(DEVICE)
            tandem_model = TandemNet(forward_model, inverse_model)
            tandem_model.load_state_dict(torch.load(self.path)['model_state_dict'])
            cie_raw, param_raw, cie_pred, param_pred = evaluate_tandem_minmax_accuracy(tandem_model, tandem_model.forward_model, self.test_loader.dataset )
            plot_cie_raw_pred(cie_raw, cie_pred) # compare the r2 sore
            plt_abs_err(cie_raw, cie_pred)  # compare the absolute mean 
            
        elif self.model_name in ['vae_hybrid']:
            cie_raw, param_raw, cie_pred, param_pred = evaluate_vae_GSNN_minmax_inverse(self.model.vae_model, self.model.forward_model, self.test_loader.dataset)
            plot_cie_raw_pred(cie_raw, cie_pred) # compare the r2 sore
            plt_abs_err(cie_raw, cie_pred)  # compare the absolute mean 
        else:
            raise NotImplementedError
            
            
    def fit_inn(self):
        temp1 = np.zeros([2,self.epochs])
        train_loader, val_loader, test_loader = get_dataloaders('tandem_net')
        loss_val_best = 1000
        
        for e in range(self.epochs):
            
            loss_train = self.train()
            loss_val = self.evaluate()
            temp1[0,e] = loss_train
            temp1[1,e] = loss_val
            print('Epoch {}, train loss {:.3f}, val loss {:.3f}'.format(
                e, loss_train, loss_val))
                                
        
        plt.plot(range(self.epochs),temp1[0,:],label='Training loss')  
        plt.plot(range(self.epochs),temp1[1,:],label='Val Loss')                      
        # plot the training and val loss VS epoches.
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        
        loss_test = self.evaluate(test=True)
        print('Training finished! Test loss {:.3f}'.format(loss_test))

        self.save_checkpoint(e, loss_test)
        print('Saved final trained model.')
    
    def save_checkpoint_e(self, epoch, loss, path_e):
        # torch.save(self.model.state_dict(), './models/'+filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path_e)
        

    def save_checkpoint(self, epoch, loss, loss_all):
        # torch.save(self.model.state_dict(), './models/'+filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'loss_all':loss_all,
        }, self.path)

    def load_checkpoint(self):

        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        loss_all = checkpoint['loss_all']
        print("Loaded model, epoch {}, loss {}..".format(epoch, loss))

# trainer class for GAN model
class GANTrainer(Trainer):

    def __init__(self,
                model,
                forward_model,
                optimizer_G,
                optimizer_D,
                train_loader,
                val_loader,
                test_loader,
                criterion,
                epochs,
                model_name,
                n_critic,
                clip_value):

        self.model_name = model_name
        self.model = model
        self.forward_model = forward_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.epochs = epochs

        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.path = './models/' + model_name + '_trained.pth'
        self.temp_path = './models/' + model_name + '_trained_temp.pth'
        self.critic = n_critic,
        self.clip_value = clip_value

    def train(self):
        self.model.train()
        loss_epoch = 0
        g_loss_epoch = 0
        d_loss_epoch = 0
        i = 1
        for x, y in self.train_loader:

            batch_size = len(x)
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(DEVICE)
            fake = torch.zeros(batch_size, 1).to(DEVICE)

            # -----------------
            #  Train Discriminator
            # -----------------

            self.optimizer_D.zero_grad()
            # generate batch of fake smaples
            z = self.model.sample_noise(batch_size).to(DEVICE)
            gen_y = self.model.generator(x, z)

            validity = self.model.discriminator(gen_y, x)
            valid = self.model.discriminator(y, x)

            d_loss = -torch.mean(valid) +torch.mean(validity)
            d_loss.backward()
            self.optimizer_D.step()

            for p in self.model.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)


            # Train the generator
            if i%5==0:
                self.optimizer_G.zero_grad()
                gen_y = self.model.generator(x, z)
                g_loss = -torch.mean(self.model.discriminator(gen_y, x))
                g_loss.backward()
                self.optimizer_G.step()
            else: 
                g_loss = 0

            g_loss_epoch += (g_loss* batch_size)
            d_loss_epoch += (d_loss* batch_size)
            i = i+1

        g_loss_epoch, d_loss_epoch = g_loss_epoch / len(self.train_loader.dataset), d_loss_epoch / len(self.train_loader.dataset)
        print('generator loss {:.6f}, discriminator loss {:.6f}'.format(g_loss_epoch, d_loss_epoch))
        return g_loss_epoch + d_loss_epoch

    def evaluate(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        with torch.no_grad():
            loss_epoch = 0
            i = 1
            for x, y in dataloader:

                batch_size = len(x)
                x, y = x.to(DEVICE), y.to(DEVICE)
                z = self.model.sample_noise(batch_size).to(DEVICE)
                gen_y = self.model.generator(x, z)

                validity = self.model.discriminator(gen_y, x)
                valid = self.model.discriminator(y, x)

                d_loss = -torch.mean(valid) +torch.mean(validity)

                # Train the generator
                if i%5==0:
                    g_loss = -torch.mean(self.model.discriminator(gen_y, x))
                else: 
                    g_loss = 0
                
                i = i+1

                loss_epoch += (g_loss + d_loss) * batch_size

            cie_raw, param_raw, cie_pred, param_pred = evaluate_gan_minmax_inverse(self.model, self.forward_model, self.val_loader.dataset, show=0)
            RMSE_cie_raw = np.sum(np.sqrt(np.average(np.square((cie_raw - cie_pred)),axis=0)))
            loss_epoch_mean = loss_epoch / len(dataloader.dataset)
        return RMSE_cie_raw


    def save_checkpoint(self, epoch, loss, loss_all, path):
        # torch.save(self.model.state_dict(), './models/'+filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'G_optimizer_state_dict': self.optimizer_G.state_dict(),
            'D_optimizer_state_dict': self.optimizer_D.state_dict(),
            'loss': loss,
            'loss_all':loss_all,
        }, path)

    def load_checkpoint(self):

        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['D_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        loss_all = checkpoint['loss_all']
        print("Loaded model, epoch {}, loss {}..".format(epoch, loss))

class GANTrainer_old(Trainer):

    def __init__(self,
                model,
                optimizer_G,
                optimizer_D,
                train_loader,
                val_loader,
                test_loader,
                criterion,
                epochs,
                model_name):

        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.epochs = epochs

        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.path = './models/' + model_name + '_trained.pth'
        self.temp_path = './models/' + model_name + '_trained_temp.pth'

    def train_old2(self):
        #self.model.train()
        loss_epoch = 0
        g_loss_epoch = 0
        d_loss_epoch = 0
        M = 100
        for x, y in self.train_loader:

            batch_size = len(x)
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = torch.cat([x]*M)
            y = torch.cat([y]*M)

            # Adversarial ground truths
            valid = torch.ones(batch_size*M, 1).to(DEVICE)
            fake = torch.zeros(batch_size*M, 1).to(DEVICE)

            # -----------------
            #  Train Generator
            # -----------------

            self.optimizer_G.zero_grad()
            self.model.generator.train()
            self.model.discriminator.eval()

            # Sample noise and labels as generator input
            z = self.model.sample_noise_M(batch_size).to(DEVICE)

            # import pdb; pdb.set_trace()
            # Generate a batch of samples
            gen_y = self.model.generator(x, z)

            # Loss measures generator's ability to fool the discriminator
            validity = self.model.discriminator(gen_y, x)
            g_loss = self.criterion(validity, valid)

            g_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()
            self.model.generator.eval()
            self.model.discriminator.train()
            # Loss for real images
            real_pred = self.model.discriminator(y, x)
            d_real_loss = self.criterion(real_pred, valid)
            d_real_loss.backward()

            # Loss for fake images
            fake_pred = self.model.discriminator(gen_y.detach(), x)
            d_fake_loss = self.criterion(fake_pred, fake)
            d_fake_loss.backward()

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            self.optimizer_D.step()

            g_loss_epoch += g_loss.to('cpu').item() * batch_size
            d_loss_epoch += d_loss.to('cpu').item() * batch_size

        g_loss_epoch, d_loss_epoch = g_loss_epoch / len(self.train_loader.dataset), d_loss_epoch / len(self.train_loader.dataset)
        print('generator loss {:.6f}, discriminator loss {:.6f}'.format(g_loss_epoch, d_loss_epoch))
        return g_loss_epoch + d_loss_epoch

    def evaluate_old2(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        with torch.no_grad():
            loss_epoch = 0
            M = 100
            for x, y in dataloader:

                batch_size = len(x)
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = torch.cat([x]*M)
                y = torch.cat([y]*M)

                # Adversarial ground truths
                valid = torch.ones(batch_size*M, 1).to(DEVICE)
                fake = torch.zeros(batch_size*M, 1).to(DEVICE)

                # Sample noise and labels as generator input
                z = self.model.sample_noise_M(batch_size).to(DEVICE)

                # Generate a batch of samples
                gen_y = self.model.generator(x, z)

                # Loss measures generator's ability to fool the discriminator
                validity = self.model.discriminator(gen_y, x)
                g_loss = self.criterion(validity, valid)


                # Loss for real images
                real_pred = self.model.discriminator(y, x)
                d_real_loss = self.criterion(real_pred, valid)

                # Loss for fake images
                fake_pred = self.model.discriminator(gen_y.detach(), x)
                d_fake_loss = self.criterion(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                # d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                loss_epoch += (g_loss.to('cpu').item() + d_loss.to('cpu').item()) * batch_size

        return loss_epoch / len(dataloader.dataset)

    def train_old(self):
        #self.model.train()
        loss_epoch = 0
        g_loss_epoch = 0
        d_loss_epoch = 0

        for x, y in self.train_loader:

            batch_size = len(x)
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(DEVICE)
            fake = torch.zeros(batch_size, 1).to(DEVICE)

            # -----------------
            #  Train Generator
            # -----------------

            self.optimizer_G.zero_grad()
            self.model.generator.train()
            self.model.discriminator.eval()

            # Sample noise and labels as generator input
            z = self.model.sample_noise(batch_size).to(DEVICE)
            print(z.size)

            # import pdb; pdb.set_trace()
            # Generate a batch of samples
            gen_y = self.model.generator(x, z)

            # Loss measures generator's ability to fool the discriminator
            validity = self.model.discriminator(gen_y, x)
            g_loss = self.criterion(validity, valid)

            g_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()
            self.model.generator.eval()
            self.model.discriminator.train()
            # Loss for real images
            real_pred = self.model.discriminator(y, x)
            d_real_loss = self.criterion(real_pred, valid)
            d_real_loss.backward()

            # Loss for fake images
            fake_pred = self.model.discriminator(gen_y.detach(), x)
            d_fake_loss = self.criterion(fake_pred, fake)
            d_fake_loss.backward()  # why backpropagate separately???

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            self.optimizer_D.step()

            g_loss_epoch += g_loss.to('cpu').item() * batch_size
            d_loss_epoch += d_loss.to('cpu').item() * batch_size

        g_loss_epoch, d_loss_epoch = g_loss_epoch / len(self.train_loader.dataset), d_loss_epoch / len(self.train_loader.dataset)
        print('generator loss {:.3f}, discriminator loss {:.3f}'.format(g_loss_epoch, d_loss_epoch))
        return g_loss_epoch + d_loss_epoch

    def evaluate_old(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        with torch.no_grad():
            loss_epoch = 0
            for x, y in dataloader:

                batch_size = len(x)
                x, y = x.to(DEVICE), y.to(DEVICE)

                # Adversarial ground truths
                valid = torch.ones(batch_size, 1).to(DEVICE)
                fake = torch.zeros(batch_size, 1).to(DEVICE)

                # Sample noise and labels as generator input
                z = self.model.sample_noise(batch_size).to(DEVICE)

                # Generate a batch of samples
                gen_y = self.model.generator(x, z)

                # Loss measures generator's ability to fool the discriminator
                validity = self.model.discriminator(gen_y, x)
                g_loss = self.criterion(validity, valid)


                # Loss for real images
                real_pred = self.model.discriminator(y, x)
                d_real_loss = self.criterion(real_pred, valid)

                # Loss for fake images
                fake_pred = self.model.discriminator(gen_y.detach(), x)
                d_fake_loss = self.criterion(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                # d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                loss_epoch += (g_loss.to('cpu').item() + d_loss.to('cpu').item()) * batch_size

        return loss_epoch / len(dataloader.dataset)


    def save_checkpoint(self, epoch, loss, loss_all, path):
        # torch.save(self.model.state_dict(), './models/'+filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'G_optimizer_state_dict': self.optimizer_G.state_dict(),
            'D_optimizer_state_dict': self.optimizer_D.state_dict(),
            'loss': loss,
            'loss_all':loss_all,
        }, path)

    def load_checkpoint(self):

        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['D_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        loss_all = checkpoint['loss_all']
        print("Loaded model, epoch {}, loss {}..".format(epoch, loss))

# Trainer class for invertible neural network
class INNTrainer(Trainer):

    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 val_loader,
                 test_loader,
                 criterion,
                 epochs,
                 model_name):

        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.epochs = epochs
        self.current_epoch = 0

        self.lambd_predict = 3
        self.lambd_latent = 400
        self.lambd_rev = 400

        self.latent_criterion = MMD_multiscale
        self.backward_criterion = MMD_multiscale

        self.path = './models/' + model_name + '_trained.pth'

    def train(self):
        self.model.train()
        loss_epoch = 0
        loss_factor = min(1., 2. * 0.002 ** (1. - (float(self.current_epoch) / self.epochs)))

        for x, y in self.train_loader:

            batch_size = len(x)

            self.optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_clean = y.clone()

            pad_x, pad_yz = self.model.create_padding(batch_size)
            pad_x, pad_yz = pad_x.to(DEVICE), pad_yz.to(DEVICE)

            y += self.model.y_noise_scale * torch.randn(batch_size, y.size(1)).float().to(DEVICE)

            x = torch.cat((x, pad_x), dim = 1)
            y = torch.cat((torch.randn(batch_size, self.model.dim_z).float().to(DEVICE), pad_yz, y), dim = 1)

            self.optimizer.zero_grad()

            output = self.model(x)[0]


            y_short = torch.cat((y[:, :self.model.dim_z], y[:, -self.model.dim_y:]), dim = 1)

    

            l = self.lambd_predict * self.criterion(output[:, self.model.dim_z:], y[:, self.model.dim_z:])

            output_block_grad = torch.cat((output[:, :self.model.dim_z],
                                           output[:, -self.model.dim_y:].data), dim=1)

            l += self.lambd_latent * self.latent_criterion(output_block_grad, y_short)
            loss_epoch += l.data.item() * len(x)

            l.backward()
            
            # Backward step:
            pad_yz = self.model.zeros_noise_scale * \
                torch.randn(batch_size, self.model.ndim_total -
                            self.model.dim_y - self.model.dim_z).to(DEVICE)
            y = y_clean + self.model.y_noise_scale * \
                torch.randn(batch_size, self.model.dim_y).to(DEVICE)

            orig_z_perturbed = (output.data[:, :self.model.dim_z] + self.model.y_noise_scale *
                                torch.randn(batch_size, self.model.dim_z).to(DEVICE))
            y_rev = torch.cat((orig_z_perturbed, pad_yz,
                            y), dim=1)
            y_rev_rand = torch.cat((torch.randn(batch_size, self.model.dim_z).to(DEVICE), pad_yz, y), dim=1)
            
            output_rev = self.model(y_rev, rev=True)[0]
            output_rev_rand = self.model(y_rev_rand, rev=True)[0]

            l_rev = (
                self.lambd_rev
                * loss_factor
                * self.backward_criterion(output_rev_rand[:, :self.model.dim_x],
                                x[:, :self.model.dim_x])
            )

            l_rev += self.lambd_predict * self.criterion(output_rev, x)
            
            loss_epoch += l_rev.data.item() * len(x)
            l_rev.backward()

            for p in self.model.parameters():
                if p.grad!=None: 
                    #print(type(p.grad))               
                    p.grad.data.clamp_(-15.00, 15.00)

            self.optimizer.step()

            # loss_epoch += loss.to('cpu').item() * len(x) + l_rev
        
        self.current_epoch += 1

        return loss_epoch / len(self.train_loader.dataset)

    def evaluate(self, test = False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader

        loss_epoch = 0
        loss_factor = 1

        with torch.no_grad():
            for x, y in dataloader:

                batch_size = len(x)

                x, y = x.to(DEVICE), y.to(DEVICE)
                y_clean = y.clone()

                pad_x, pad_yz = self.model.create_padding(batch_size)
                pad_x, pad_yz = pad_x.to(DEVICE), pad_yz.to(DEVICE)

                y += self.model.y_noise_scale * torch.randn(batch_size, y.size(1)).float().to(DEVICE)

                x = torch.cat((x, pad_x), dim = 1)
                y = torch.cat((torch.randn(batch_size, self.model.dim_z).float().to(DEVICE), pad_yz, y), dim = 1)

                output = self.model(x)[0]

                y_short = torch.cat((y[:, :self.model.dim_z], y[:, -self.model.dim_y:]), dim = 1)

                l = self.lambd_predict * self.criterion(output[:, self.model.dim_z:], y[:, self.model.dim_z:])

                output_block_grad = torch.cat((output[:, :self.model.dim_z],
                                            output[:, -self.model.dim_y:].data), dim=1)

                l += self.lambd_latent * self.latent_criterion(output_block_grad, y_short)
                loss_epoch += l.data.item() * len(x)
                
                # Backward step:
                pad_yz = self.model.zeros_noise_scale * \
                    torch.randn(batch_size, self.model.ndim_total -
                                self.model.dim_y - self.model.dim_z).to(DEVICE)
                y = y_clean + self.model.y_noise_scale * \
                    torch.randn(batch_size, self.model.dim_y).to(DEVICE)

                orig_z_perturbed = (output.data[:, :self.model.dim_z] + self.model.y_noise_scale * torch.randn(
                    batch_size, self.model.dim_z).to(DEVICE))
                y_rev = torch.cat((orig_z_perturbed, pad_yz, y), dim=1)
                y_rev_rand = torch.cat((torch.randn(batch_size, self.model.dim_z).to(DEVICE), pad_yz, y), dim=1)
                
                output_rev = self.model(y_rev, rev=True)[0]
                output_rev_rand = self.model(y_rev_rand, rev=True)[0]

                l_rev = (
                    self.lambd_rev
                    * loss_factor
                    * self.backward_criterion(output_rev_rand[:, :self.model.dim_x],
                                    x[:, :self.model.dim_x])
                )

                l_rev += self.lambd_predict * self.criterion(output_rev, x)
                
                loss_epoch += l_rev.data.item() * len(x)

                for p in self.model.parameters():
                    if p.grad!=None:
                        p.grad.data.clamp_(-15.00, 15.00)
            # loss_epoch += loss.to('cpu').item() * len(x) + l_rev

        return loss_epoch / len(dataloader.dataset)
