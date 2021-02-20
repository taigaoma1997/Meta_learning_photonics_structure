import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# class Trainer():

#     def __init__(self,
#                  model,
#                  optimizer,
#                  train_loader,
#                  val_loader,
#                  test_loader,
#                  criterion,
#                  epochs,
#                  model_name):

#         self.model_name = model_name
#         self.model = model
#         self.optimizer = optimizer
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.test_loader = test_loader
#         self.criterion = criterion
#         self.epochs = epochs

#         self.path = './models/' + model_name + '.pth'

#     def train(self):
#         self.model.train()
#         loss_epoch = 0
#         for x, y in self.train_loader:

#             self.optimizer.zero_grad()
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             pred = self.model(x, y)
#             loss = self.get_loss(x, y, pred)
#             loss.backward()
#             self.optimizer.step()
#             loss_epoch += loss.to('cpu').item() * len(x)

#         return loss_epoch / len(self.train_loader.dataset)

#     def val(self):
#         self.model.eval()
#         loss_epoch = 0
#         with torch.no_grad():
#             for x, y in self.val_loader:

#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 pred = self.model(x, y)
#                 loss = self.get_loss(x, y, pred)
#                 loss_epoch += loss.to('cpu').item() * len(x)

#         return loss_epoch / len(self.val_loader.dataset)

#     def test(self):
#         self.model.eval()
#         loss_epoch = 0
#         with torch.no_grad():
#             for x, y in self.test_loader:

#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 pred = self.model(x, y)
#                 loss = self.get_loss(x, y, pred)
#                 loss_epoch += loss.to('cpu').item() * len(x)

#         return loss_epoch / len(self.test_loader.dataset)

#     def fit(self):

#         for e in range(self.epochs):

#             loss_train = self.train()
#             loss_val = self.val()
#             print('Epoch {}, train loss {:.3f}, val loss {:.3f}'.format(
#                 e, loss_train, loss_val))

#         loss_test = self.test()
#         print('Training finished! Test loss {:.3f}'.format(loss_test))

#         self.save_checkpoint(e, loss_test)
#         print('Saved final trained model.')

#     def save_checkpoint(self, epoch, loss):
#         # torch.save(self.model.state_dict(), './models/'+filename)

#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': loss,
#         }, self.path)

#     def load_checkpoint(self):

#         checkpoint = torch.load(self.path)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch = checkpoint['epoch']
#         loss = checkpoint['loss']

#         print("Loaded model, epoch {}, loss {}..".format(epoch, loss))

#     def get_loss(self, x, y, pred):
#         '''
#         Loss for training simple forward and inverse networks.
#         '''
        
#         if self.model_name in ['inverse_model', 'forward_model']:
#             return self.criterion(pred, y)
#         elif self.model_name in ['tandem_net']:
#             return self.criterion(pred, x)
#         elif self.model_name == 'vae':
#             # see Appendix B from VAE paper:
#             # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#             # https://arxiv.org/abs/1312.6114
#             # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#             recon_x, mu, logvar, y_pred = pred
#             recon_loss = self.criterion(recon_x, x)
#             KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#             pred_loss = self.criterion(y_pred, y)

#             # print(BCE, KLD)
#             return recon_loss + KLD + pred_loss
#         else:
#             raise NotImplementedError


def evaluate_tandem_accuracy(model, dataset):
    '''
    returns:
        x_raw: original desired xyY
        x_raw_pred: xyY predicted by the forward module for the inversely designed structure
        y_raw: original structure parameters
        y_raw_pred: inversely designed parameters.
    '''
    model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # get MSE for the design
        y_pred = model.pred(x)
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Tandem net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Tandem Design RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = model(x, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()


def evaluate_simple_inverse(forward_model, inverse_model, dataset):

    inverse_model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # get MSE for the design
        y_pred = inverse_model(x, y)
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

def evaluate_vae_inverse(forward_model, vae_model, configs, dataset):

    vae_model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        mu, logvar = torch.zeros((len(x), configs['latent_dim'])), torch.zeros((len(x), configs['latent_dim']))
        z = vae_model.reparameterize(mu, logvar).to(DEVICE)
        y_pred = vae_model.decode(z, x)

        # get MSE for the design
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

def evaluate_gan_inverse(forward_model, gan_model, configs, dataset):

    gan_model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        z = gan_model.sample_noise(len(x)).to(DEVICE)
        y_pred = gan_model.generator(x, z)

        # get MSE for the design
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()


def evaluate_inn_inverse(forward_model, model, configs, dataset):
    '''
    The dataset need to be in inverse format, i.e., x corresponds to target while y corresponds to design.
    '''

    model.eval()

    mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)

    def infer_design(model, dataset):
        x, y = dataset.y.to(DEVICE), dataset.x.to(DEVICE)
        x_dim = x.size()[1]
        y_clean = y.clone()

        batch_size = len(x)
        pad_x, pad_yz = model.create_padding(batch_size)
        pad_x = pad_x.to(DEVICE)
        pad_yz = pad_yz.to(DEVICE)

        y += model.y_noise_scale * torch.randn(batch_size, y.size(1)).float().to(DEVICE)
        y = torch.cat((torch.randn(batch_size, model.dim_z).float().to(DEVICE), pad_yz, y), dim = 1)

        y = y_clean + model.y_noise_scale * torch.randn(batch_size, model.dim_y).to(DEVICE)

        y_rev_rand = torch.cat((torch.randn(batch_size, model.dim_z).to(DEVICE), pad_yz, y), dim=1)
        output_rev_rand = model(y_rev_rand, rev=True)
        x_pred = output_rev_rand[:, :model.dim_x]

        return x_pred
    
    with torch.no_grad():
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        y_pred = infer_design(model, dataset)

        # get MSE for the design
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()


def count_params(model):

    return sum([np.prod(layer.size()) for layer in model.parameters() if layer.requires_grad])

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# multiscale MMD loss
def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(DEVICE),
                  torch.zeros(xx.shape).to(DEVICE),
                  torch.zeros(xx.shape).to(DEVICE))

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)