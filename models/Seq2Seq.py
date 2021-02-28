import torch
import random
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.dropout = params.seq2seq['dropout']
        self.input_size = params.seq2seq['input_size']
        self.hidden_size = params.seq2seq['hidden_size']
        self.num_layers = params.seq2seq['num_layers']
        
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True
            , bidirectional=True)

    def forward(self, src):
        out, (hidden, cell ) = self.rnn(src)
        return out, hidden, cell 

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.dropout = params.seq2seq['dropout']
        self.input_size = params.seq2seq['input_size']
        self.hidden_size = params.seq2seq['hidden_size']
        self.num_layers = params.seq2seq['num_layers']
        self.output = params.seq2seq['output']

        self.rnn = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True
            , bidirectional=True)
        
        self.fc1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)

    def forward(self, input_enc, hidden, cell):
        
        input_enc = input_enc.unsqueeze(1)
        out, (hidden, cell ) = self.rnn(input_enc, (hidden , cell))
        return self.fc1(out.squeeze(1)), hidden, cell

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, params):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = params.batch_size
        self.n_win = params.dataset['n_win']
        self.dim = params.dataset['dim']
        self.hidden_size = params.seq2seq['hidden_size']
        self.device = params.device
        self.output = params.seq2seq['output']
        self.fc1 = nn.Linear(self.hidden_size*2, self.output)
        
        assert encoder.hidden_size == decoder.hidden_size, "hidden dimensions of encoder and decoder do not match"
        assert encoder.num_layers == decoder.num_layers, "number of layers of encode and decoder do not match"

    def forward(self, src, teacher_forcing_ratio = 0.5):

        outputs = torch.zeros(src.shape[0], self.n_win, self.hidden_size*2).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_out, hidden , cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        #input_enc = trg[:,0,:]
        #input_enc = torch.zeros(src.shape[0],self.hidden_size*2).to(self.device)
        input_dec = enc_out[:,0,:] 

        for t in range(1, self.n_win):

            # take the last hidden and cell state to produce output and 
            # next hidden and cell state
            output, hidden, cell = self.decoder(input_dec, hidden, cell)
            
            outputs[:,t,:] = output

            # decide whether to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use the actual next label as next input
            # if not, use the predicted input

            input_dec = enc_out[:,t,:] if teacher_force else output
        
        return self.fc1(outputs)