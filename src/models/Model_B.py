import torch
from torch.autograd import Variable as V
from src.models.Model_A import model_A
from src.utils.decorators import timer
from src.utils.modelUtil import split_batch
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import branchLoss, modelOuts, RnnResults
import snoop
import pdb

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
            if self.params.geography: out_rnn_chunk=square_normalize(out_rnn_chunk)
            out_rnn_list.append(out_rnn_chunk)
            loss_main_chunk=self.criterion(out_rnn_chunk*cp_mask_chunk, batch_label_chunk*cp_mask_chunk)
            loss_main_list.append(loss_main_chunk.item())
            sample_size=cp_mask_chunk.sum()
            loss_main_chunk /=sample_size
            if self.params.cp_predict:
                assert self.params.cp_detect, "cp detection is not true while cp prediction is true"
                cp_logits, loss_cp = self._changePointNet(vec_64, target=cps_chunk)
                loss_main_chunk +=loss_cp/(cps_chunk.shape[0]*cps_chunk.shape[1])
            loss_main_chunk.backward()
            # after doing back prob, detach rnn state to implement TBPTT
            # now rnn_state was detached and chain of gradients was broken
            rnn_state = self.model['lstm']._detach_rnn_state(rnn_state)
            
        loss_main=sum(loss_main_list)
        out_rnn=torch.cat(out_rnn_list, 1).detach()
        vec_64=torch.cat(vec64_list, 1).detach()
        return RnnResults(out=out_rnn, out_nxt=vec_64, loss_main=loss_main)
        
    def _rnn(self, x):
        vec_64, out_rnn, _ = self.model['lstm'](x)
        return RnnResults(out=out_rnn, out_nxt=vec_64)
    
    def _rnnNet(self, x, **kwargs):
        target=kwargs.get('target')
        mask=kwargs.get('mask')
        if self.enable_tbptt:
            rnnResults = self._tbtt(x, target, mask)
            return rnnResults
        else:
            rnnResults = self._rnn(x)
            if self.params.geography: rnnResults.out=square_normalize(rnnResults.out)
            # loss_main=self.criterion(out_rnn*mask, target.coord_main*mask) if target is not None else None
            return rnnResults
        
    def _getLossInner(self, outs, target):
        auxLoss=self.criterion(outs.coord_aux, target.coord_main)
        mainLoss=self.criterion(outs.coord_main, target.coord_main)
        return branchLoss(loss_main=mainLoss, loss_aux=auxLoss)

    def _inner(self,x,**kwargs):
        target=kwargs.get('target')
        mask = kwargs.get('mask')
        if mask is None: mask=torch.ones((x.shape[0],self.params.n_win,1), dtype=torch.uint8, device=self.params.device)
        self.enable_tbptt=False
        if self.params.tbptt and any([m.training for m in list(self.model.values())]):
            # print("{}bling tbtt".format("ena" if self.params.tbptt and any([m.training for m in list(self.model.values())]) else "disa"))
            self.enable_tbptt=True
        out_aux, x_nxt = self._auxNet(x)
        if self.params.geography: out_aux=square_normalize(out_aux)
        rnnResults = self._rnnNet(x_nxt, target=target, mask=mask)
        outs = modelOuts(coord_main = rnnResults.out*mask, coord_aux=out_aux*mask)
        if self.enable_tbptt:
            return outs, rnnResults.out_nxt, rnnResults.loss_main
        return outs, rnnResults.out_nxt

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
