import torch
from torch import nn

class Conv1d(nn.Module):
    def __init__(self, params):
        super(Conv1d, self).__init__()
        self.dropout = nn.Dropout(p=params.conv['dropout'])
        #self.layernorm = nn.LayerNorm(params.conv['input'])
        #self.fc = nn.Linear(params.conv['input'], params.conv['hidden_unit'])
        self.relu = nn.ReLU()
        
        self.conv1d = nn.Conv1d(params.conv['hidden_unit'], params.conv['output'], params.conv['kernel_size'], padding=params.conv['padding'], padding_mode=params.conv['padding_mode'])

    def forward(self, x):
        
        #x_permuted = self.fc(x).permute(0,2,1)
        x_permuted = x.permute(0,2,1)
        out = self.conv1d(self.relu(self.dropout(x_permuted)))
        out_1 = out.permute(0,2,1)
        return out_1