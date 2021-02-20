import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        '''
        layer_sizes: list of input sizes
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, output_size)])

    def forward(self, x, y):
        return self.net(x)


class TandemNet(nn.Module):

    def __init__(self, forward_model, inverse_model):
        super(TandemNet, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(self, x, y):
        '''
        Pass the desired target x to the tandem network.
        '''

        pred = self.inverse_model(x, y)
        out = self.forward_model(pred, None)

        return out

    def pred(self, x):
        pred = self.inverse_model(x, None)
        return pred


class cVAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim=256, forward_dim=3):
        super(cVAE, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()])

        self.forward_net = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(hidden_dim),
                                           nn.Linear(hidden_dim, forward_dim)])

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):

        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):

        h = self.encoder(x)
        y_pred = self.forward_net(h)

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, y_pred

# conditional GAN


class Generator(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=128):
        super(Generator, self).__init__()

        self.input_size = input_size

        self.net = nn.Sequential(*[nn.Linear(input_size + noise_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, output_size)])

    def forward(self, x, noise):
        y = self.net(torch.cat((x, noise), dim=1))
        return y


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU()])

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.aux_layer = nn.Sequential(nn.Linear(128, 3))

    def forward(self, x):
        h = self.net(x)
        validity = self.adv_layer(h)
        # label = self.aux_layer(h)

        return validity


class cGAN(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=128):
        super(cGAN, self).__init__()

        self.generator = Generator(
            input_size, output_size, noise_dim=noise_dim, hidden_dim=hidden_dim)
        self.discriminator = Discriminator(
            output_size, input_size, hidden_dim=hidden_dim)

        self.noise_dim = noise_dim

    def forward(self, x, noise):

        y_fake = self.generator(x, noise)
        validity = self.discriminator(y_fake)

        return validity

    def sample_noise(self, batch_size):

        z = torch.tensor(np.random.normal(
            0, 1, (batch_size, self.noise_dim))).float()
        return z

# invertible neural network

class INN(nn.Module):
    def __init__(self, ndim_total, dim_x, dim_y, dim_z, hidden_dim = 128):
        super(INN, self).__init__()

        nodes = [InputNode(ndim_total, name = 'input')]
        self.hidden_dim = hidden_dim
        self.ndim_total = ndim_total
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        for k in range(4):
            nodes.append(Node(nodes[-1],
                                GLOWCouplingBlock,
                                {'subnet_constructor': self.subnet_fc, 'clamp': 2.0},
                                name=F'coupling_{k}'))

            nodes.append(Node(nodes[-1],
                            PermuteRandom,
                            {'seed': k},
                            name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name = 'output'))

        self.model = ReversibleGraphNet(nodes, verbose = False)
        self.zeros_noise_scale = 5e-2
        self.y_noise_scale = 1e-1

    def forward(self, x, rev=False):
        return self.model(x, rev=rev)

    def subnet_fc(self, c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, self.hidden_dim),
                             nn.ReLU(),
                             nn.Linear(self.hidden_dim, c_out))

    def create_padding(self, batch_size):

        pad_x = self.zeros_noise_scale * torch.randn(batch_size, self.ndim_total - self.dim_x)
        pad_yz = self.zeros_noise_scale * torch.randn(batch_size, self.ndim_total - self.dim_y - self.dim_z)

        return pad_x, pad_yz

    # def add_noise_y(self, y, batch_size):

    #     y += self.y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)