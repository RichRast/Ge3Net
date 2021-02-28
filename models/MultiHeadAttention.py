import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class attention(nn.Module):
    def __init__(self, params):
        super(attention, self).__init__() # Initialize self._modules as OrderedDict
        self.key_size = params.att['key_size']
        self.query_size = params.att['query_size']
        self.value_size = params.att['value_size']
        self.linear_keys = nn.Linear(params.att['input_size'], self.key_size)
        self.linear_query = nn.Linear(params.att['input_size'], self.query_size)
        self.Linear_value = nn.Linear(params.att['input_size'], self.value_size)
        self.sqrt_key_size = math.sqrt(self.key_size)
        self.dropout = nn.Dropout(p=params.att['dropout'])
        self.layernorm = nn.LayerNorm(params.att['input_size'] + params.att['value_size'])

    def forward(self, x):
        # shape of x is BxTxemb_dim
        
        keys = self.linear_keys(x) #shape BxTxkey_size
        query = self.linear_query(x) #shape BxTXquery_size
        value = self.Linear_value(x) #shape BxTxvalue_size

        num = torch.bmm(query, keys.permute(0,2,1)) # shape BxTxT
        weight = F.softmax(num/self.sqrt_key_size, dim =2)
        att_score = torch.bmm(weight, value)
        out1 = self.dropout(self.layernorm(x + att_score))
        out = torch.cat((x, out1), dim=2)
        return out, att_score, weight