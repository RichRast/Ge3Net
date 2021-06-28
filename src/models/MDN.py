import torch
import torch.nn as nn

class MixtureDensityNetwork(nn.Module):
    def __init__(self, params, input_size, output_size):
        super(MixtureDensityNetwork, self).__init__()
        self.num_gaussian= params.mdn_num_gaussian
        self.hidden = params.mdn_hidden
        self.input_size = input_size
        self.output_size = output_size

        # self.pi = 
        # self.mu = 
        # self.sigma = 
        pass

    def forward(self, x):
        pass
