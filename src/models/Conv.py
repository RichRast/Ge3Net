import torch
from torch import nn
from src.models.modelParamsSelection import Selections

class Conv1d(nn.Module):
    def __init__(self, params):
        super(Conv1d, self).__init__()
        self.params=params
        self.option=Selections.get_selection()
        self.dropout = nn.Dropout(p= self.params.conv_dropout)
        self.normalizationlayer1 = self.option['normalizationLayer'][self.params.conv_net_norm](self.params.conv_hidden_unit)
        self.relu = nn.ReLU()
        
        self.conv1d = nn.Conv1d( self.params.conv_hidden_unit,  self.params.conv_out,  self.params.conv_kernel_size, padding= self.params.conv_padding, padding_mode= self.params.conv_padding_mode)

    def forward(self, x):
        x_permuted = x.permute(0,2,1)
        out = self.conv1d(self.normalizationlayer1(self.dropout(self.relu(x_permuted))))
        out_1 = out.permute(0,2,1)
        return out_1