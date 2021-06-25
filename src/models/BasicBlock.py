import torch
from torch import nn
from src.models.modelParamsSelection import Selections

class basicBlock(nn.Module):
    def __init__(self, params):
        super(basicBlock, self).__init__()
        self.dropout = nn.Dropout(p=params.basic_Block_dropout)
        self.layernorm = nn.LayerNorm(params.basic_Block_in)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(params.basic_Block_in, params.basic_Block_out)

    def forward(self, x):
        out = self.dropout(self.relu(self.layernorm(self.fc(x))))
        return out
    
class logits_Block(nn.Module):
    def __init__(self, params, input):
        super(logits_Block, self).__init__()
        self.dropout = nn.Dropout(p=params.logits_Block_dropout)
        self.layernorm = nn.LayerNorm(params.logits_Block_hidden)
        self.relu = nn.ReLU()
        self.input = input

        self.fc1 = nn.Linear(self.input, params.logits_Block_hidden)
        self.fc2 = nn.Linear(params.logits_Block_hidden, params.logits_Block_hidden1)
        self.layernorm1 = nn.LayerNorm(params.logits_Block_hidden1)
        self.fc3 = nn.Linear(params.logits_Block_hidden1, params.logits_Block_out)

    def forward(self, x):
        out_1 = self.dropout(self.relu(self.layernorm(self.fc1(x))))
        out_2 = self.dropout(self.relu(self.layernorm1(self.fc2(out_1))))
        logits = self.fc3(out_2)
        return logits

    def loss(self, loss_fn, logits, target):
        out_loss = loss_fn(logits, target)
        return out_loss

class mlp(nn.Module):
    def __init__(self, params, input):
        super(mlp, self).__init__()
        self.option=Selections.get_selection()
        self.params=params
        self.input=input
        self.dropout = nn.Dropout(p=params.mlp_dropout)
        self.normalizationLayer1 = self.option['normalizationLayer'][self.params.mlp_net_norm](self.params.mlp_net_hidden)
        self.relu = nn.LeakyReLU(params.leaky_relu_slope)
        self.fc1 = nn.Linear(self.input, self.params.mlp_net_hidden)
        self.fc2 = nn.Linear(self.mlp_net_hidden, self.params.mlp_net_out)

    def forward(self, x):
        out1 = self.normalizationLayer1(self.fc1(x))
        out = self.fc2(self.relu(self.dropout(out1)))
        return out1, out

