import torch
from torch.autograd import Variable as V
from src.models.Model_A import model_A
from src.utils.decorators import timer
from src.utils.modelUtil import split_batch
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import t_accr, t_out, t_results

class model_E(model_A):
    _network=['aux', 'conv', 'cp']
    def __init__(self, *args, params):
        super().__init__(*args, params=params)
        
    def _inner(self,x,**kwargs):
        target=kwargs.get('target')
        mask = kwargs.get('mask')
        if mask is None:
            mask=torch.ones_like(target.coord_main, dtype=float)
        out_aux, x_nxt = self._auxNet(x)
        if self.params.geography: out_aux=square_normalize(out_aux)
        out_conv = self.model['conv'](x)
        if self.params.geography: out_conv=square_normalize(out_conv)
        outs = t_out(coord_main = out_conv*mask, coord_aux=out_aux*mask)
        if target is not None:
            loss_aux = self.criterion(out_aux*mask, target.coord_main*mask)
            loss_main = self.criterion(out_conv*mask, target.coord_main*mask)
            loss_inner = t_accr(loss_aux=loss_aux, loss_main=loss_main)
        return outs, loss_inner

    def _outer(self, x, target, mask):
        outs, x_nxt, loss_inner = self._inner(x, target=target, mask=mask)
        sample_size=mask[...,0].sum()
        lossBack=loss_inner.loss_aux/sample_size
        accr=t_accr(loss_aux=loss_inner.loss_aux.item())
        if self.params.cp_predict:
            cp_logits, cp_accr = self._changePointNet(x_nxt, target=target.cp_logits)
            outs=outs._replace(cp_logits=cp_logits)
            cp_accr_detached=cp_accr._replace(loss_cp=cp_accr.loss_cp.items())
        else:
            cp_accr_detached=None
        if not self.enable_tbptt:
            lossBack+= loss_inner.loss_main
            accr=accr._replace(loss_main=loss_inner.loss_main.item())
        if not self.enable_tbptt and self.params.cp_predict: 
            lossBack+=cp_accr.loss_cp/(target.cp_logits.shape[0]*target.cp_logits.shape[1])
        lossBack.backward()
        return outs, t_results(t_accr=accr, t_cp_accr=cp_accr_detached)
