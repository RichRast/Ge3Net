import torch
from torch import nn
import numpy as np

class BiRNN(nn.Module):
    def __init__(self, params, input_size, rnn='lstm'):
        super(BiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = params.rnn_net_hidden
        self.num_layers = params.rnn_net_n_layers
        self.output = params.rnn_net_out
        self.dropout = params.rnn_net_dropout
        self.rnn = rnn
        self.tbptt = params.tbptt
        self.device = params.device
        self.bidirectional = params.rnn_net_bidirectional
        assert self.rnn in ['lstm', 'gru'], 'rnn type is not supported'

        if self.rnn == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True
            , bidirectional=self.bidirectional)
        elif self.rnn == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True
                              , bidirectional=self.bidirectional)

        self.fc1 = nn.Linear(self.hidden_size * (1+1*self.bidirectional), self.output)

    def repackage_rnn_state(self):
        self.rnn_state = self._detach_rnn_state(self.rnn_state)

    def _detach_rnn_state(self, h):
        # Wraps hidden states in new Tensors, to detach them from their history.
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return h[0].detach(), h[1].detach()

    def forward(self, x, rnn_state=None):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * (1+1*self.bidirectional), x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * (1+1*self.bidirectional), x.size(0), self.hidden_size).to(self.device)

        # to supress this warning when using data parallel
        # RNN module weights are not part of single contiguous chunk of memory. This means they need to be 
        # compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
        # self.rnn.flatten_parameters()
        # truncated backprop - detach the state after n steps and use it for the next sequence
        out1, rnn_state = self.rnn(x, rnn_state)
        out = self.fc1(out1)
        return out1, out, rnn_state

class lstm(nn.Module):
    def __init__(self, params, input_size, rnn='lstm'):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = params.rnn_net_hidden
        self.num_layers = params.rnn_net_n_layers
        self.output = params.rnn_net_out
        self.dropout = params.rnn_net_dropout
        self.rnn = rnn
        self.tbptt = params.tbptt
        self.device = params.device
        self.bidirectional = params.rnn_net_bidirectional
        
        assert self.rnn in ['lstm', 'gru'], 'rnn type is not supported'
        if self.num_layers >1:
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True
            , bidirectional=self.bidirectional)
        else:
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True
            , bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layernorm = nn.LayerNorm(self.hidden_size*(1+1*self.bidirectional))
        self.fc1 = nn.Linear(self.hidden_size*(1+1*self.bidirectional), self.output)
        
    def forward(self, x, rnn_state=None):
        
        out1, rnn_state = self.rnn(x, rnn_state)
        out1 = self.layernorm(self.dropout(out1))

        out = self.fc1(out1)
        return out1, out, rnn_state
