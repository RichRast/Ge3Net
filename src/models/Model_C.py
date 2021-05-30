import torch
from torch.autograd import Variable as V
from src.models.Model_B import model_B
from src.utils.decorators import timer
from src.utils.modelUtil import split_batch
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import modelOuts

class model_C(model_B):
    _network=['base', 'lstm', 'cp']
    def __init__(self, *args, params):
        super().__init__(*args, params=params)
            
    def _baseNet(self, x):
        out1 = self.model['base'](x)
        aux_diff = get_gradient(out1)
        out_nxt = torch.cat((out1, aux_diff), dim =2)
        return out1, out_nxt

    def _inner(self,x,**kwargs):
        target=kwargs.get('target')
        mask = kwargs.get('mask')
        if mask is None: mask=1
        self.enable_tbptt=False
        if self.params.tbptt and any([m.training for m in list(self.model.values())]):
            # print("{}bling tbtt".format("ena" if self.params.tbptt and any([m.training for m in list(self.model.values())]) else "disa"))
            self.enable_tbptt=True
        out_aux, x_nxt = self._baseNet(x)
        if self.params.geography: out_aux=square_normalize(out_aux)
        rnnResults = self._rnnNet(x_nxt, target=target, mask=mask)
        outs = modelOuts(coord_main = rnnResults.out*mask, coord_aux=out_aux*mask)
        if self.enable_tbptt:
            return outs, rnnResults.out_nxt, rnnResults.loss_main
        return outs, rnnResults.out_nxt
