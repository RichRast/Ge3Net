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
    nll = - torch.log(torch.sum(prob, dim=2))
    return torch.mean(nll)

def sample(pi, sigma, mu):
    """
    Draw sample from mixture of gaussians parametrized by pi, sigma and mu
    Arguments:
        input:

        output:

    """
    # Choose the gaussian to pick the sample from
    pis = Categorical(pi).sample().view(pi.shape(0),1,1,1)
    gaussian_noise = torch.randn((sigma.shape[3], sigma.shape[0], sigma.shape[1]), requires_grad=False)
    variance_samples= sigma.gather(2, pis).detach().squeeze()
    mean_samples=mu.detach().gather(2, pis).squeeze()
    return (gaussian_noise*variance_samples + mean_samples).permute(1,2,0).contiguous()