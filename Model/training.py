import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
from utils import MMD_multiscale
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from models import MLP, TandemNet, cVAE, cGAN, INN
from utils import evaluate_simple_inverse, evaluate_tandem_accuracy, evaluate_vae_inverse, evaluate_gan_inverse, evaluate_inn_inverse
from configs import get_configs
from plotting_utils import compare_cie_dist, compare_param_dist, plot_cie, plot_cie_raw_pred
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

    def train(self):
        self.model.train()
        loss_epoch = 0
        for x, y in self.train_loader:

            self.optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = self.model(x, y)
            loss = self.get_loss(x, y, pred)
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.to('cpu').item() * len(x)

        return loss_epoch / len(self.train_loader.dataset)

    def evaluate(self, test = False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        loss_epoch = 0
        with torch.no_grad():
            for x, y in dataloader:

                x, y = x.to(DEVICE), y.to(DEVICE)
                if self.model_name == 'vae_GSNN':
                    pred = self.model(x,y)
                elif self.model_name == 'vae_Full':
                    pred = self.model(x,y)
                else:
                    pred = self.model(x,y)
                loss = self.get_loss(x, y, pred)
                loss_epoch += loss.to('cpu').item() * len(x)

        return loss_epoch / len(dataloader.dataset)

    def fit(self):
        temp1 = np.zeros([2,self.epochs]);
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
        
    def fit_inn(self):
        temp1 = np.zeros([2,self.epochs]);
        train_loader, val_loader, test_loader = get_dataloaders('tandem_net')
        for e in range(self.epochs):
            
            loss_train = self.train()
            loss_val = self.evaluate()
            temp1[0,e] = loss_train
            temp1[1,e] = loss_val
            print('Epoch {}, train loss {:.3f}, val loss {:.3f}'.format(
                e, loss_train, loss_val))
            if e%50 == 0:
                plt.plot(range(self.epochs),temp1[0,:],label='Training loss')  
                plt.plot(range(self.epochs),temp1[1,:],label='Val Loss')                      
                # plot the training and val loss VS epoches.
                plt.xlabel('epochs')
                plt.ylabel('Loss')
                plt.title('weight_decay: 1e-5 + learning_rate 5e-4 + Epochs %i' %e)
                plt.ylim([0, 100])
                plt.xlim([1, e])
                plt.legend()
                plt.show()
                
                loss_test = self.evaluate(test=True)
                path_e  = './models/inn_trained_epoch_' + str(e) + '.pth'
                self.save_checkpoint_e(e, loss_test, path_e)
                
                configs = get_configs('inn')
                model_temp = INN(configs['ndim_total'], configs['input_dim'], configs['output_dim'], dim_z = configs['latent_dim']).to(DEVICE)
                model_temp.load_state_dict(torch.load(path_e)['model_state_dict'], strict=False)
                forward_model_temp = MLP(4, 3).to(DEVICE)
                forward_model_temp.load_state_dict(torch.load('./models/forward_model_trained.pth')['model_state_dict'])
                cie_raw, param_raw, cie_pred, param_pred = evaluate_inn_inverse(forward_model_temp, model_temp, configs, test_loader.dataset)
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
                fig.savefig('./models/inn_trained_epoch_' + str(e) + '.png')
                                
        
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
        

    def save_checkpoint(self, epoch, loss):
        # torch.save(self.model.state_dict(), './models/'+filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.path)

    def load_checkpoint(self):

        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Loaded model, epoch {}, loss {}..".format(epoch, loss))

    def get_loss(self, x, y, pred):
        '''
        Loss for training simple forward and inverse networks.
        '''
        
        if self.model_name in ['inverse_model', 'forward_model']:
            return self.criterion(pred, y)
        elif self.model_name in ['tandem_net']:
            return self.criterion(pred, x)
        elif self.model_name == 'vae':
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            recon_x, mu, logvar, y_pred = pred
            recon_loss = self.criterion(recon_x, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = self.criterion(y_pred, y)

            # print(BCE, KLD)
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
            recon_x, mu, logvar, y_pred = pred
            recon_loss = self.criterion(recon_x, x)
            
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = self.criterion(y_pred, y)
            
            return recon_loss + KLD+5*pred_loss        
        
        else:
            raise NotImplementedError

# trainer class for GAN model
class GANTrainer(Trainer):

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
        
    def train(self):
        self.model.train()
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

            # Sample noise and labels as generator input
            z = self.model.sample_noise(batch_size).to(DEVICE)

            # import pdb; pdb.set_trace()
            # Generate a batch of samples
            gen_y = self.model.generator(x, z)

            # Loss measures generator's ability to fool the discriminator
            validity = self.model.discriminator(gen_y)
            g_loss = self.criterion(validity, valid)

            g_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # Loss for real images
            real_pred = self.model.discriminator(y)
            d_real_loss = self.criterion(real_pred, valid)

            # Loss for fake images
            fake_pred = self.model.discriminator(gen_y.detach())
            d_fake_loss = self.criterion(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            g_loss_epoch += g_loss.to('cpu').item() * batch_size
            d_loss_epoch += d_loss.to('cpu').item() * batch_size

        g_loss_epoch, d_loss_epoch = g_loss_epoch / len(self.train_loader.dataset), d_loss_epoch / len(self.train_loader.dataset)
        print('generator loss {:.3f}, discriminator loss {:.3f}'.format(g_loss_epoch, d_loss_epoch))
        return g_loss_epoch + d_loss_epoch

    def evaluate(self, test=False):
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
                validity = self.model.discriminator(gen_y)
                g_loss = self.criterion(validity, valid)


                # Loss for real images
                real_pred = self.model.discriminator(y)
                d_real_loss = self.criterion(real_pred, valid)

                # Loss for fake images
                fake_pred = self.model.discriminator(gen_y.detach())
                d_fake_loss = self.criterion(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                # d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                loss_epoch += (g_loss.to('cpu').item() + d_loss.to('cpu').item()) * batch_size

        return loss_epoch / len(dataloader.dataset)

    def save_checkpoint(self, epoch, loss):
        # torch.save(self.model.state_dict(), './models/'+filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'G_optimizer_state_dict': self.optimizer_G.state_dict(),
            'D_optimizer_state_dict': self.optimizer_D.state_dict(),
            'loss': loss,
        }, self.path)

    def load_checkpoint(self):

        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['D_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

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

            output = self.model(x)

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
            
            output_rev = self.model(y_rev, rev=True)
            output_rev_rand = self.model(y_rev_rand, rev=True)

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

                output = self.model(x)

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
                
                output_rev = self.model(y_rev, rev=True)
                output_rev_rand = self.model(y_rev_rand, rev=True)

                l_rev = (
                    self.lambd_rev
                    * loss_factor
                    * self.backward_criterion(output_rev_rand[:, :self.model.dim_x],
                                    x[:, :self.model.dim_x])
                )

                l_rev += self.lambd_predict * self.criterion(output_rev, x)
                
                loss_epoch += l_rev.data.item() * len(x)

                for p in self.model.parameters():
                    p.grad.data.clamp_(-15.00, 15.00)
            # loss_epoch += loss.to('cpu').item() * len(x) + l_rev

        return loss_epoch / len(dataloader.dataset)
