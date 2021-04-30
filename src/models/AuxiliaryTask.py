import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models import BasicBlock

def swish(x):
    return x * F.sigmoid(x)

class AuxNetwork(nn.Module):
    def __init__(self, params):
        super(AuxNetwork, self).__init__()
        self.batch_size = params.batch_size
        self.input = params.aux_net_in
        self.output = params.aux_net_out
        self.hidden_unit = params.aux_net_hidden
        self.hidden_unit1 = params.aux_net_hidden1
        self.hidden_unit2 = params.aux_net_hidden2
        self.n_win = params.n_win
        self.linears = nn.ModuleList([nn.Linear(self.input, self.hidden_unit) for _ in range(self.n_win)])
        self.dropout = nn.Dropout(p=params.aux_net_dropout)
        self.layernorm = nn.LayerNorm(self.hidden_unit)
        self.relu = nn.LeakyReLU(params.leaky_relu_slope)
        self.aux_net_block = params.aux_net_block
        self.fc2 = nn.Linear(self.hidden_unit, self.hidden_unit1)
        self.layernorm2 = nn.LayerNorm(self.hidden_unit1)
        self.dropout1 = nn.Dropout(p=params.aux_net_dropout1)

        if self.aux_net_block:
            self.n_layers = params.aux_net_n_layers
            layers = []
            for _ in range(self.n_layers):
                layers.append(nn.Linear(self.hidden_unit1, self.hidden_unit1))
                layers.append(nn.LayerNorm(self.hidden_unit1))
                layers.append(nn.LeakyReLU(params.leaky_relu_slope))
                layers.append(nn.Dropout(p=params.aux_net_dropout2))
            self.block_layers = nn.Sequential(*layers)

        self.fc3 = nn.Linear(self.hidden_unit1, self.hidden_unit2)
        self.layernorm3 = nn.LayerNorm(self.hidden_unit2)
        self.fc4 = nn.Linear(self.hidden_unit2, self.output)
        self.device = params.device

    def forward(self, x):
        out_1 = torch.zeros([x.shape[0], self.hidden_unit, self.n_win]).to(self.device)

        for i in range(self.n_win):
            out_1[:, 0:self.hidden_unit, i] = self.linears[i](x[:, i * self.input:(i + 1) * self.input].clone())

        out_1 = self.dropout(out_1)  # shape 256x100x317

        out_1 = out_1.permute(0, 2, 1)  # shape 256x317x100
        out_1 = out_1.contiguous()
        out_1 = out_1.view(-1, self.hidden_unit)  # shape (256*317)x100
        out_1 = self.layernorm(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.dropout1(self.relu(self.layernorm2(self.fc2(out_1))))
        if self.aux_net_block:
            out_2 = self.block_layers(out_2)
        out_3 = self.dropout(self.relu(self.layernorm3(self.fc3(out_2))))
        out_4 = self.fc4(out_3)  # 81152, 3

        out_1 = out_1.view(x.shape[0], self.n_win, self.hidden_unit)  # shape 256x317x100

        out_2 = out_2.view(x.shape[0], self.n_win, self.hidden_unit1)  # shape 256x317x100
        
        out_3 = out_3.view(x.shape[0], self.n_win, self.hidden_unit2)  # shape 256x317x50

        out_4 = out_4.view(x.shape[0], self.n_win, self.output)

        return out_1, out_2, out_3, out_4


