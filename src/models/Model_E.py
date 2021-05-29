import torch
from torch.autograd import Variable as V
from src.models.Model_A import model_D
from src.utils.dataUtil import square_normalize
from src.main.evaluation import modelOuts, branchLoss

class model_E(model_D):
    _network=['aux', 'conv', 'cp']
    def __init__(self, *args, params):
        super().__init__(*args, params=params)
        
    def _inner(self,x,**kwargs):
        mask = kwargs.get('mask')
        out_aux, x_nxt = self._auxNet(x)
        if self.params.geography: out_aux=square_normalize(out_aux)
        out_conv = self.model['conv'](x_nxt)
        outs = modelOuts(coord_main = out_conv*mask, coord_aux=out_aux*mask)
        return outs, out_conv

    def _outer(self, x, target, mask):
        innerResults= self._inner(x, target=target, mask=mask)
        if self.enable_tbptt:
            outs, out_nxt, loss_main = innerResults
        else:
            outs, out_nxt = innerResults
            loss_main_backprop = self.criterion(outs.coord_main, target.coord_main)
        sample_size=mask.sum()
        loss_aux = self.criterion(outs.coord_aux, target.coord_main)
        lossBack=loss_aux/sample_size
        if not self.enable_tbptt:
            lossBack+= loss_main_backprop
            loss_main = loss_main_backprop.item()
        loss_inner = branchLoss(loss_aux=loss_aux.item(), loss_main=loss_main)
        if self.params.cp_predict:
            cp_logits, loss_cp = self._changePointNet(out_nxt, target=target.cp_logits)
            loss_inner.loss_cp=loss_cp.item()
            outs.cp_logits=cp_logits
        if not self.enable_tbptt and self.params.cp_predict: 
            lossBack+=loss_cp/(target.cp_logits.shape[0]*target.cp_logits.shape[1])
        return outs, loss_inner, lossBack