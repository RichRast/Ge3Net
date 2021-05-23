import torch
from src.models.Model_B import model_B
from src.utils.dataUtil import get_gradient


class model_C(model_B):
    _network=['aux', 'lstm', 'cp']
    def __init__(self, *args, params):
        super().__init__(*args, params=params)
            
    def _auxNet(self, x):
        out1, _, _, out4 = self.model['aux'](x)
        out1 = out1.reshape(x.shape[0], self.params.n_win, self.params.aux_net_hidden)
        # add residual connection by taking the gradient of aux network predictions
        aux_diff = get_gradient(out4)
        x_nxt = torch.cat((out4, aux_diff), dim =2)
        return out4, x_nxt



    
    
    

    

    
            