import torch
from torch import nn

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
    def __init__(self, params):
        super(logits_Block, self).__init__()
        self.dropout = nn.Dropout(p=params.logits_Block_dropout)
        self.layernorm = nn.LayerNorm(params.logits_Block_hidden)
        #self.layernorm = nn.LayerNorm(params.logits_Block_in)
        self.relu = nn.ReLU()
        if params.model=='Model_D':
            self.input = params.rnn_net_hidden * (1+1*params.rnn_net_bidirectional)
        elif params.model=='Model_K':
            self.input = params.att1_value_size 
        elif params.model=='Model_A':
            self.input = params.aux_net_hidden + params.dataset_dim
        #self.fc = nn.Linear(params.logits_Block_in, params.logits_Block_out)
        self.fc1 = nn.Linear(self.input, params.logits_Block_hidden)
        self.fc2 = nn.Linear(params.logits_Block_hidden, params.logits_Block_hidden1)
        self.layernorm1 = nn.LayerNorm(params.logits_Block_hidden1)
        self.fc3 = nn.Linear(params.logits_Block_hidden1, params.logits_Block_out)

    def forward(self, x):
        out_1 = self.dropout(self.relu(self.layernorm(self.fc1(x))))
        out_2 = self.dropout(self.relu(self.layernorm1(self.fc2(out_1))))
        logits = self.fc3(out_2)
        #logits = self.fc(x)
        return logits

class Multi_Block(nn.Module):
    def __init__(self, params):
        super(Multi_Block, self).__init__()
        if params.model=='Model_D':
            self.input = params.rnn_net_hidden * (1+1*params.rnn_net_bidirectional)
        elif params.model=='Model_K':
            self.input = params.att1_value_size 
        elif params.model=='Model_A':
            self.input = params.aux_net_hidden + params.dataset_dim
        self.fc1 = nn.Linear(self.input, params.Multi_Block_out)
        
    def forward(self, x):
        logits = self.fc1(x)
        return logits

class Residual_Block(nn.Module):
    def __init__(self, params):
        super(Residual_Block, self).__init__()
        self.dropout = nn.Dropout(p=params.Residual_Block_dropout)
        self.layernorm = nn.LayerNorm(params.Residual_Block_hidden)
        self.relu = nn.ReLU()
        if params.model=='Model_D':
            self.input = params.rnn_net_hidden * (1+1*params.rnn_net_bidirectional)
        elif params.model=='Model_K':
            self.input = params.att1_value_size 
        elif params.model=='Model_A':
            self.input = params.aux_net_hidden + params.dataset_dim
        self.fc1 = nn.Linear(self.input, params.Residual_Block_out)
        # self.fc1 = nn.Linear(self.input, params.Residual_Block_hidden)
        # self.fc2 = nn.Linear(params.Residual_Block_hidden, params.Residual_Block_hidden1)
        # self.layernorm1 = nn.LayerNorm(params.Residual_Block_hidden1)
        # self.fc3 = nn.Linear(params.Residual_Block_hidden1, params.Residual_Block_out)

        

    def forward(self, x):
        # out_1 = self.dropout(self.relu(self.layernorm(self.fc1(x))))
        # out_2 = self.dropout(self.relu(self.layernorm1(self.fc2(out_1))))
        # logits = self.fc3(out_2)
        logits = self.fc1(x)
        
        return logits 