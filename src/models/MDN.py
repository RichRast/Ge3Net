import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
import pdb

# credit to https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

class MixtureDensityNetwork(nn.Module):
    """
    MDN layer where the output can be mapped to a mixture of distributions- gaussian here

    Arguments:
        input: 
            x: (BXWxGXD): B is the batch size, W is the number of windows, G is the number of gaussians
            and D is the input dimension vector
        output:
            pi: (BXWXG)
            sigma: (BXWXGXO)
            mu: (BXWXGXO)
            B is batch size, W is number of windows, G is the number of gaussians and O is the output
            dimension vector
    """
    def __init__(self, params, input_size, output_size):
        super(MixtureDensityNetwork, self).__init__()
        self.params = params
        self.num_gaussian = params.mdn_num_gaussian
        self.hidden = params.mdn_hidden
        self.input_size = input_size
        self.output_size = output_size

        self.pi = nn.Sequential(
            nn.Linear(self.input_size, self.num_gaussians),
            nn.Softmax(dim=2)
        )
        self.mu = nn.Linear(self.input_size, self.output_size*self.num_gaussian)
        self.sigma = nn.Linear(self.input_size, self.output_size*self.num_gaussian)
        
    def forward(self, x):
        pi = self.pi(x)
        sigma = torch.exp(self.sigma(x)) # sigma >=0 , page 274, eq 5.151 Bishop
        sigma = sigma.view(-1, self.params.n_win, self.num_gaussian, self.output_size)
        mu = self.mu(x)
        mu = mu.view(-1, self.params.n_win, self.num_gaussian, self.output_size)
        return pi, sigma, mu

def gaussian_probability(sigma, mu, x):
    """
    Given an input x that is the output of MDN layer, return the probability of x
    belonging to a gaussian with parameters mu, sigma
    Arguments:
        input: 
            sigma: (BXWXGXO)
            mu: (BXWXGXO)
            x: (BXWXGXO)
        output:
            probabilities: (BXWXG): probability of each window for the prospective pi's
            parameterized by mu and sigma
    """
    prob_per_pi = ONEOVERSQRT2PI*torch.exp(-0.5*torch.pow((mu-x),2))/sigma
    return torch.prod(prob_per_pi, 3)

def mdn_loss(pi, sigma, mu, x):
    """
    compute negative log likelihood
    """
    prob = pi * gaussian_probability(sigma, mu, x)
    nll = - torch.log()

def sample(pi, sigma, mu):
    ...