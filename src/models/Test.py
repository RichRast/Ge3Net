from models.Model_A import model_A
import torch
import numpy as np
from collections import namedtuple
import os
import sys
sys.path.insert(1, os.environ.get('USER_PATH'))
from src.utils.decorators import timer
from src.utils.modelUtil import activate_mc_dropout
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss, \
    gradient_reg, eval_cp_batch, t_accr, t_out, t_cp_accr, t_results, Running_Average

class model_B(model_A):
    _network=['aux', 'main', 'cp']
    def __init__(self, *args, params):
        super().__init__(*args, params)
        
    def _rnnNet(self, x):
        if self.params.tbptt:
            vec64, out_rnn, _= self._tbtt(x)
        else:
            vec64, out_rnn, _ = self._rnn(x)
        return out_rnn, vec64 
        
    def _auxNet(self, x):
        out1, _, _, out4 = self.model['aux'](x)
        out1 = out1.reshape(x.shape[0], self.params.n_win, self.params.aux_net_hidden)
        # add residual connection by taking the gradient of aux network predictions
        aux_diff = get_gradient(out4)
        x_nxt = torch.cat((out1, aux_diff), dim =2)
        return out4, x_nxt

    def _inner(self,x):
        out_aux, x_nxt = self._auxNet(x)
        if self.params.geography: square_normalize(out_aux)

        out_rnn, vec64 = self._rnnNet(x)
        outs = t_out(coord = out_aux)
        return outs, x_nxt

    def _getLoss(self, y, target, **kwargs):
        mask=kwargs.get('mask')
        runAvgObj=kwargs.get('runAvgObj')
        cpRunAvgObj=kwargs.get('cpRunAvgObj')
        if mask is None:
            mask=torch.ones_like(target.coord, dtype=float)
        weighted_loss = self.criterion(y.coord*mask, target.coord*mask)
        loss_sum=weighted_loss
        if self.params.cp_predict: 
            cp_loss= self.BCEwithLogits(y.cp_logits, target.cp_logits)
            loss_sum += cp_loss
            batchCpLoss=t_cp_accr(cp_loss=cp_loss)
        # back propogate loss1 + loss 2
        sample_size=mask.sum()
        lossBack = loss_sum/sample_size
        #calculate accuracy/loss to report
        batchLoss=t_accr(weighted_loss=weighted_loss)
        
        batchAvg, batchCpAvg = self._evaluateAccuracy(y, target, sample_size=sample_size, batchLoss=batchLoss, \
            batchCpLoss=batchCpLoss, runAvgObj=runAvgObj, cpRunAvgObj=cpRunAvgObj)
        return (batchAvg, batchCpAvg), lossBack

    def _outer(self, x, target, mask, **kwargs):
        runAvgObj=kwargs.get('runAvgObj')
        cpRunAvgObj=kwargs.get('cpRunAvgObj')
        outs, x_nxt = self._inner(x)
        if self.params.cp_predict: 
            cp_logits = self._changePointNet(x_nxt)
            outs=outs._replace(cp_logits=cp_logits)
        loss, lossBack = self._getLoss(outs, target, mask=mask, runAvgObj=runAvgObj, cpRunAvgObj=cpRunAvgObj) 
        lossBack.backward()
        return outs, loss

    def _changePointNet(self, x):
        cp_pred_logits = self.model['cp'](x)
        return cp_pred_logits
    
    def _evaluateAccuracy(self, y, target, **kwargs):
        runAvgObj=kwargs.get('runAvgObj')
        cpRunAvgObj=kwargs.get('cpRunAvgObj')
        batchLoss=kwargs.get('batchLoss')
        batchCpLoss=kwargs.get('batchCpLoss')
        sample_size=kwargs.get('sample_size')

        # calculate loss per batch
        cp_pred_logits=y.cp_logits
        cp_target=target.cp_logits
        cp_pred = (torch.sigmoid(cp_pred_logits)>0.5).int()
        cp_target=cp_target.squeeze(2)
        cp_pred=cp_pred.squeeze(2)
        precision, recall, _, _, balanced_accuracy = eval_cp_batch(cp_target, cp_pred)
        
        batchLoss=batchLoss._replace(l1_loss=self.L1Loss(y.coord, target.coord).item(),\
            loss_aux=self.L1Loss(y.coord, target.coord).item(),\
            mse_loss= self.MseLoss(y.coord, target.coord).item(),\
            smoothl1_loss= self.smoothL1Loss(y.coord, target.coord, self.params.device),\
            )
        batchCpLoss=batchCpLoss._replace(Precision=precision,\
            Recall=recall,\
            BalancedAccuracy=balanced_accuracy)
        
        # update the running avg object
        for key, val in runAvgObj.items():
            if getattr(batchLoss,key) is not None:
                val.update(getattr(batchLoss,key), sample_size)
            
        
        cpRunAvgObj['cp_loss'].update(batchCpLoss.cp_loss, target.cp_logits.shape[0]*target.cp_logits.shape[1])
        cpRunAvgObj['Precision'].update(batchCpLoss.Precision, 1)
        cpRunAvgObj['Recall'].update(batchCpLoss.Recall, 1)
        cpRunAvgObj['BalancedAccuracy'].update(batchCpLoss.BalancedAccuracy, 1)
        
        # get the running average for batches in this epoch so far by calling the 
        # running avg object       
        batchAvg=t_accr(l1_loss=runAvgObj.get('l1_loss')(), \
            loss_aux=runAvgObj.get('loss_aux')(), \
            mse_loss=runAvgObj.get('mse_loss')(), \
            smoothl1_loss=runAvgObj.get('smoothl1_loss')(), \
            weighted_loss=runAvgObj.get('weighted_loss')(), \
            )
        batchCpAvg=t_cp_accr(cp_loss=cpRunAvgObj.get('cp_loss'),\
            Precision=cpRunAvgObj.get('Precision'),\
            Recall=cpRunAvgObj.get('Recall'),\
            BalancedAccuracy=cpRunAvgObj.get('Balanced_Accuracy'))

        del batchLoss, batchCpLoss
        torch.cuda.empty_cache()
        return batchAvg, batchCpAvg

    def _getFromMcSamples(self, outs_list):
        cat_outs = torch.cat(outs_list, 0).contiguous()
        mean_outs = cat_outs.view(self.params.mc_samples, -1, self.params.n_win, cat_outs.shape[-1]).mean(0)
        var_outs = cat_outs.view(self.params.mc_samples, -1, self.params.n_win, cat_outs.shape[-1]).var(0)
        return t_out(coord=mean_outs, y_var=var_outs)

    def _getSample(self,**kwargs):
        idx=kwargs.get('idx')
        data_vcf_idx=kwargs.get('vcf_idx')
        target=kwargs.get('label')
        y=kwargs.get('out')
        target_idx = target.coord[idx,...].detach().cpu().numpy().reshape(-1, self.params.dataset_dim)
        y_idx = y.coord[idx,:,:self.params.n_comp_overall].detach().cpu().numpy().reshape(-1, self.params.n_comp_overall)
        if self.params.superpop_predict:
            y_sp_idx = y.sp[idx,:].detach().cpu().numpy().reshape(1,-1)
        vcf_idx = data_vcf_idx[idx,:].detach().cpu().numpy().reshape(-1, 1)
        return y_idx, target_idx, vcf_idx

    def _plotSample(self, wandb, plotObj, *kwargs):
        idx=kwargs.get('idx')
        phase=kwargs.get('phase')
        idxSample = kwargs.get('idxSample')
        idxLabel = kwargs.get('idxLabel')
        vcf_idx= kwargs.get('idxVcf_idx')
        fig, ax = plotObj.plot_index(idxSample, idxLabel, vcf_idx)
        wandb.log({f" Image for idx {idx} for {phase}":wandb.Image(fig)})

    def _logger(self, wandb, **kwargs):
        batch_num=kwargs.get('batch_num')
        batchAvg=kwargs.get('batchAvg')
        batchCpAvg=kwargs.get('batchCpAvg')
        phase=kwargs.get('phase')
        wandb.log({f"MainTask_Loss/{phase}":batchAvg.l1_loss, "batch_num":batch_num})
        wandb.log({f"AuxTask_Loss/{phase}":batchAvg.loss_aux, "batch_num":batch_num})
        wandb.log({f"cp_loss/{phase}":batchCpAvg.cp_loss,"batch_num":batch_num})
        wandb.log({f"cp_accr/{phase}":batchCpAvg._asdict(), "batch_num":batch_num})
        if self.params.residual:
            wandb.log({f"residual_loss/{phase}":batchAvg.residual_loss, "batch_num":batch_num})
            