import torch.nn as nn

class transformer(nn.Module):
    def __init__(self, params):
        super(transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(params.transf['input_size'], params.transf['num_heads'])
        self.encoder = nn.TransformerEncoder(self.encoder_layer, params.transf['n_layers'])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(params.transf['dropout'])
        self.layernorm = nn.LayerNorm(params.transf['hidden_size'])
        self.fc1 = nn.Linear(params.transf['input_size'], params.transf['hidden_size'])
        self.fc2 = nn.Linear(params.transf['hidden_size'], params.transf['output'])

    def forward(self, x):
        out1 = self.encoder(x)
        out2 = self.dropout(self.activation(self.layernorm(self.fc1(out1))))
        out3 =  self.fc2(out2)
        return out1, out3