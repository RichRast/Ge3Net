import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
# credit to http://nlp.seas.harvard.edu/2018/04/03/attention.html

class attention_single(nn.Module):
    def __init__(self, params, input_size):
        super(attention_single, self).__init__() # Initialize self._modules as OrderedDict
        self.input_size = input_size
        self.key_size = input_size
        self.query_size = input_size
        self.value_size = input_size
        
        self.linear_keys = nn.Linear(self.input_size, self.key_size)
        self.linear_query = nn.Linear(self.input_size, self.query_size)
        self.Linear_value = nn.Linear(self.input_size, self.value_size)
        self.sqrt_key_size = math.sqrt(self.key_size)
        self.dropout = nn.Dropout(p=params.att_dropout)
        self.layernorm = nn.LayerNorm(self.value_size)
        self.beta = params.att_beta

    def forward(self, x):
        # shape of x is BxTxemb_dim
        keys = self.linear_keys(x) #shape BxTxkey_size
        query = self.linear_query(x) #shape BxTXquery_size
        value = self.Linear_value(x) #shape BxTxvalue_size

        num = torch.bmm(query, keys.permute(0,2,1)) # shape BxTxT
        weight = F.softmax((self.beta*num)/self.sqrt_key_size, dim =2)
        att_score = torch.bmm(weight, value)
        out = x + self.dropout(att_score)
        out1 = self.layernorm(out)
        return out1, att_score, weight       
        
class PositionalEncoding(nn.Module):
    def __init__(self, params, d_model):
        super(PositionalEncoding, self).__init__()
        """
        Computes the positional encoding of input x, given by
        PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        where
        pos: position of the sequence window, [0, seq_len)
        i: embedding dim idx, [0,d_model/2)
        d_model: embed dim
        pe: positional encoding of the embed dim , shape[batch_size, embed_dim, 1]
        """
        PE_constant = params.PE_constant
        self.d_model= d_model if d_model%2==0 else d_model + 1
        self.dropout = nn.Dropout(params.PE_dropout)
        pos = torch.arange(0, params.n_win, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, d_model/2, dtype=torch.float)
        den_term = torch.exp(math.log(PE_constant)* (2*i/d_model))
        pe = torch.zeros(params.n_win, self.d_model)
        pe[:,0::2] = torch.sin(pos/den_term)
        pe[:,1::2] = torch.cos(pos/den_term)
        pe = pe.unsqueeze(2).permute(2,0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[...,:-1] if x.shape[-1]%2!=0 else self.pe)
        return self.dropout(x)


class FFNN(nn.Module):
    def __init__(self, params, input1_size, input2_size, input3_size, output_size):
        super(FFNN, self).__init__()
        self.input1_size= input1_size
        self.input2_size= input2_size
        self.input3_size= input3_size
        self.output_size=output_size
        self.fc1 = nn.Linear(self.input1_size, self.input2_size)
        self.dropout1 = nn.Dropout(params.FFNN_dropout1)
        self.layernorm1 = nn.LayerNorm(self.input2_size)
        self.fc2 = nn.Linear(self.input2_size, self.input3_size)

        self.dropout2 = nn.Dropout(params.FFNN_dropout2)
        self.layernorm2 = nn.LayerNorm(self.input3_size)
        self.fc3 = nn.Linear(self.input3_size, self.output_size)
        if params.FFNN_activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        out1 = x + self.fc1(x)
        out1 = self.layernorm1(self.activation(self.dropout1(out1)))
        out2 = x + self.fc2(out1)
        out2 = self.layernorm2(self.activation(self.dropout2(out2)))
        out3 = self.fc3(out2)
        return out2, out3

class AttentionBlock(nn.Module):
    def __init__(self, params, input1_size, input2_size, input3_size, output_size):
        super(AttentionBlock, self).__init__()
        self.input1_size=input1_size
        self.input2_size= input2_size
        self.input3_size= input3_size
        self.output_size=output_size
        self.attention = attention_single(params, self.input1_size)
        self.ffnn = FFNN(params, self.input1_size, self.input2_size, self.input3_size, output_size)
    
    def forward(self,x):
        out_nxt, _, weight = self.attention(x)
        _, out_att = self.ffnn(out_nxt)
        return out_att
