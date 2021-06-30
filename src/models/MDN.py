import torch
import torch.nn as nn
import pdb

# credit to https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py

class MixtureDensityNetwork(nn.Module):
    def __init__(self, params, input_size, output_size):
        super(MixtureDensityNetwork, self).__init__()
        self.num_gaussian = params.mdn_num_gaussian
        self.hidden = params.mdn_hidden
        self.input_size = input_size
        self.output_size = output_size

        self.pi = nn.Sequential(
            nn.Linear(self.input_size, self.num_gaussians),
            nn.Softmax(dim=1)
        )
        self.mu = nn.Linear(self.input_size, self.output_size*self.num_gaussian)
        self.sigma = nn.Linear(self.input_size, self.output_size*self.num_gaussian)
        
    def forward(self, x):
        pi = self.pi(x)
        sigma = torch.exp(self.sigma(x)) # sigma >=0 , page 274, eq 5.151
        sigma = sigma.view(-1, self.num_gaussian, self.output_size)
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussian, self.output_size)
        return pi, sigma, mu

def gaussian_probability(sigma, mu, target):
    ...

def mdn_loss(pi, sigma, mu, target):
    ...

def sample(pi, sigma, mu):
    ...