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
        self.beta = params.att['beta']

    def forward(self, x):
        # shape of x is BxTxemb_dim
        
        keys = self.linear_keys(x) #shape BxTxkey_size
        query = self.linear_query(x) #shape BxTXquery_size
        value = self.Linear_value(x) #shape BxTxvalue_size

        num = torch.bmm(query, keys.permute(0,2,1)) # shape BxTxT
        weight = F.softmax((self.beta*num)/self.sqrt_key_size, dim =2)
        att_score = torch.bmm(weight, value)
        out = torch.cat((x, att_score), dim=2)
        out1 = self.layernorm(self.dropout(out))
        return out1, att_score, weight
    
class attention_single(nn.Module):
    def __init__(self, params):
        super(attention_single, self).__init__() # Initialize self._modules as OrderedDict
        self.key_size = params.att1_key_size
        self.query_size = params.att1_query_size
        self.value_size = params.att1_value_size
        self.input = params.aux_net_hidden + params.dataset_dim
        self.linear_keys = nn.Linear(self.input, self.key_size)
        self.linear_query = nn.Linear(self.input, self.query_size)
        self.Linear_value = nn.Linear(self.input, self.value_size)
        self.sqrt_key_size = math.sqrt(self.key_size)
        self.dropout = nn.Dropout(p=params.att1_dropout)
        self.layernorm = nn.LayerNorm(self.value_size)
        self.beta = params.att1_beta

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
        

class MultiHeadedAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadedAttention, self).__init__()
        # 4 because 3 linear layers are for Q, K and V and 
        # the last linear layer is for final W matrix to 
        # get the attention score with softmax in the desired shape
        self.seq_len = params.dataset['n_win']
        self.device = params.device
        self.num_heads = params.att['num_heads']
        #self.attn = nn.ModuleList([attention_single(params) for _ in range(self.num_heads)])
        self.attn = _get_clones(attention_single(params), self.num_heads)
        self.fc = nn.Linear(params.att['input_size'], params.att['z_f_size'])
        self.layernorm = nn.LayerNorm(params.att['z_f_size'])
        self.dropout = nn.Dropout(params.att['dropout2'])
        
    def forward(self, x):
        batch_size = x.shape[0]
        weight = torch.zeros([self.num_heads, batch_size, self.seq_len, self.seq_len]).to(self.device)
        for i in range(self.num_heads):
            out1, att_score, weight[i,:,:,:] = self.attn[i](x)
            if i >0:
                att_score_f = torch.cat((att_score_f, att_score), dim=2)
            else:
                att_score_f = att_score
                return out1, att_score, weight
               
        z_f = att_score_f
        out2 = x + self.dropout(z_f)
        out3 = self.layernorm(out2)
        return out3, att_score_f, weight

class PositionalEncoding(nn.Module):
    def __init__(self, params):
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
        d_model = params.d_model
        self.dropout = nn.Dropout(params.PE_dropout)
        pos = torch.arange(0, params.n_win, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, d_model/2, dtype=torch.float)
        den_term = torch.exp(math.log(PE_constant)* (2*i/d_model))
        pe = torch.zeros(params.n_win, params.d_model)
        pe[:,0::2] = torch.sin(pos/den_term)
        pe[:,1::2] = torch.cos(pos/den_term)
        pe = pe.unsqueeze(2).permute(2,0,1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class FFNN(nn.Module):
    def __init__(self, params):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(params.FFNN_input1, params.FFNN_input2)
        self.dropout1 = nn.Dropout(params.FFNN_dropout1)
        self.layernorm1 = nn.LayerNorm(params.FFNN_input2)
        self.fc2 = nn.Linear(params.FFNN_input2, params.FFNN_input3)
        self.dropout2 = nn.Dropout(params.FFNN_dropout2)
        self.layernorm2 = nn.LayerNorm(params.FFNN_input3)
        self.fc3 = nn.Linear(params.FFNN_input3, params.FFNN_output)
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

