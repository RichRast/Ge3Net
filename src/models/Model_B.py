import torch
from torch.autograd import Variable as V
import numpy as np
from collections import namedtuple
import os
import sys
sys.path.insert(1, os.environ.get('USER_PATH'))
from src.models.Model_A import model_A
from src.utils.decorators import timer
from src.utils.modelUtil import split_batch
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss, \
    gradient_reg, eval_cp_batch, t_accr, t_out, t_cp_accr, t_results, Running_Average

class model_B(model_A):
    _network=['aux', 'lstm', 'cp']
    def __init__(self, *args, params):
        super().__init__(*args, params=params)
        
    def _tbtt(self, x, target, mask):
        rnn_state = None
        bptt_batch_chunks = split_batch(x.clone(), self.params.tbptt_steps)
        batch_cps_chunks = split_batch(target.cp_logits, self.params.tbptt_steps)
        batch_label = split_batch(target.coord_main, self.params.tbptt_steps)
        loss_main_list, out_rnn_list, vec64_list=[],[],[]
        for x_chunk, batch_label_chunk, cps_chunk in zip(bptt_batch_chunks, batch_label, batch_cps_chunks):
            x_chunk = V(x_chunk, requires_grad=True)
            vec_64, out_rnn_chunk, rnn_state = self.model['lstm'](x_chunk, rnn_state)
            vec64_list.append(vec_64)
            cp_mask_chunk = (cps_chunk==0).float()
            if self.params.geography: square_normalize(out_rnn_chunk)
            out_rnn_list.append(out_rnn_chunk)
            loss_main_chunk=self.criterion(out_rnn_chunk*cp_mask_chunk, batch_label_chunk*cp_mask_chunk)
            loss_main_list.append(loss_main_chunk.item())
            if self.params.cp_predict:
                assert self.params.cp_detect, "cp detection is not true while cp prediction is true"
                cp_logits, accr_cp = self._changePointNet(vec_64, target=cp_mask_chunk)
                loss_main_chunk +=accr_cp.loss_cp
            loss_main_chunk.backward()
            # after doing back prob, detach rnn state to implement TBPTT
            # now rnn_state was detached and chain of gradients was broken
            rnn_state = self.model['lstm']._detach_rnn_state(rnn_state)
            
        loss_main=sum(loss_main_list)
        out_rnn=torch.cat(out_rnn_list, 1).detach()
        vec_64=torch.cat(vec64_list, 1).detach()
        return out_rnn, vec_64, loss_main
        
    def _rnn(self, x):
        vec_64, out_rnn, _ = self.model['lstm'](x)
        return  out_rnn, vec_64

    def _rnnNet(self, x, **kwargs):
        target=kwargs.get('target')
        mask=kwargs.get('mask')
        loss_main=None
        if self.enable_tbptt:
            out_rnn, vec_64, loss_main= self._tbtt(x, target, mask)
        else:
            out_rnn, vec_64 = self._rnn(x)
            if self.params.geography: square_normalize(out_rnn)
            if target is not None: loss_main=self.criterion(out_rnn*mask, target.coord_main*mask)

        return out_rnn, vec_64, loss_main
        
    def _inner(self,x,**kwargs):
        target=kwargs.get('target')
        mask = kwargs.get('mask')
        if mask is None:
            mask=torch.ones_like(target.coord_main, dtype=float)
        if self.params.tbptt and any([m.training for m in list(self.model.values())]):
            # print("{}bling tbtt".format("ena" if self.params.tbptt and any([m.training for m in list(self.model.values())]) else "disa"))
            self.enable_tbptt=True
        else:
            self.enable_tbptt=False
        out_aux, x_nxt = self._auxNet(x)
        if self.params.geography: square_normalize(out_aux)
        out_rnn, vec64, loss_main = self._rnnNet(x_nxt, target=target, mask=mask)
        outs = t_out(coord_main = out_rnn, coord_aux=out_aux)
        if target is not None:
            loss_aux = self.criterion(out_aux*mask, target.coord_main*mask)
            loss_inner = t_accr(loss_aux=loss_aux, loss_main=loss_main)
        return outs, vec64, loss_inner

    def _outer(self, x, target, mask):
        outs, x_nxt, loss_inner = self._inner(x, target=target, mask=mask)
        lossBack=loss_inner.loss_aux

        if self.params.cp_predict: 
            cp_logits, cp_accr = self._changePointNet(x_nxt, target=target.cp_logits)
            outs=outs._replace(cp_logits=cp_logits)
        if not self.enable_tbptt:
            lossBack+=cp_accr.cp_loss
            lossBack+= loss_inner.loss_main
        lossBack.backward()
        return outs, t_results(t_accr=t_accr(loss_main=loss_inner.loss_main, loss_aux=loss_inner.loss_aux, weighted_loss=loss_inner.loss_main), t_cp_accr=cp_accr)

    
    
    

    

    
            