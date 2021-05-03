import torch
import numpy as np
from collections import namedtuple
from utils.decorators import timer
from utils.modelUtil import activate_mc_dropout, \
    square_normalize, get_gradient, Running_Average
from main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss, \
    gradient_reg, eval_cp_batch, t_accr, t_out, t_cp_accr, t_results

class Ge3Net(object):
    _network=['aux', 'cp']
    def __init__(self, *args, params):
        self.model={}
        for k,m in zip(self._network, args):
            self.model[k]=m
        self.params = params
        self._init_loss()
        
    def _init_loss(self):
        if self.params.geography:  
            self.criterion = GcdLoss() 
        else: 
            self.criterion = Weighted_Loss(reduction='sum', alpha=self.params.weightLoss_alpha)
        self.smoothL1Loss = SmoothL1Loss(reduction='sum', beta=self.params.SmoothLoss_beta)
        self.L1Loss = torch.nn.L1Loss(reduction='sum')
        self.MseLoss = torch.nn.MSELoss(reduction='sum')
        self.BCEwithLogits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        
    @timer
    def train(self, optimizer, training_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        trainRunAvgObj = {k:Running_Average() for k in t_accr}
        trainCpRunAvgObj = {k:Running_Average() for k in t_cp_accr}
        
        for l, m in self.model.items():
            m.train()

        for i, train_gen in enumerate(training_generator):
            train_x, train_y, vcf_idx, cps, _ = train_gen
            train_x = train_x[:, :self.params.chmlen].float().to(self.params.device)
            train_y = train_y.to(self.params.device)
            cps = cps.to(self.params.device)
            cp_mask = (cps==0).float() # mask for transition windows
            train_labels = t_out(coord=train_y, cp_logits=cps)
            
            # Forward pass
            # update the gradients to zero
            optimizer.zero_grad()

            train_outs, (trainBatchAvg, trainCpBatchAvg) = self._outer(train_x, train_labels, \
                mask=cp_mask, runAvgObj=trainRunAvgObj, cpRunAvgObj=trainCpRunAvgObj)
            
            # update the weights
            optimizer.step()
            
            #logging
            if wandb:
                self._logger(wandb, batchAvg=trainBatchAvg, batchCpAvg=trainCpBatchAvg, batch_num=i)
                idx = np.random.choice(train_x.shape[0],1)
                idxSample, idxLabel = self._getSample(out=train_outs, label=train_labels, vcf_idx=vcf_idx, idx=idx)
                if plotObj is not None: self._plotSample(wandb, plotObj, idxSample=idxSample, \
                    idxLabel=idxLabel, idx=idx, phase="train")
            del train_x, train_y, cps, train_outs
    
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=trainBatchAvg, t_cp_accr=trainCpBatchAvg, t_out=train_outs)

    @timer
    def valid(self, validation_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        valRunAvgObj = {k:Running_Average() for k in t_accr}
        valCpRunAvgObj = {k:Running_Average() for k in t_cp_accr}

        for l, m in self.model.items():
            m.eval()
 
        with torch.no_grad():
            for i, val_gen in enumerate(validation_generator):
                val_x, val_y, vcf_idx, cps, _ = val_gen
                val_x = val_x[:, 0:self.params.chmlen].float().to(self.params.device)
                val_y = val_y.to(self.params.device)
                cps = cps.to(self.params.device)
                val_labels = t_out(coord=val_y, cp_logits=cps)

                if self.params.mc_dropout:
                    activate_mc_dropout(*list(self.model.values()))
                else:
                    assert self.params.mc_samples==1, "MC dropout disabled"

                val_outs_list, x_nxt_list=[],[]
                for _ in range(self.params.mc_samples):
                    val_outs_tmp, x_nxt_tmp = self._inner(val_x)
                    val_outs_list.append(val_outs_tmp)
                    x_nxt_list.append(x_nxt_tmp)
                    
                val_outs = self._getFromMcSamples(val_outs_list)
                x_nxt = self._getFromMcSamples(x_nxt_list)
                if self.params.cp_predict: 
                    cp_logits = self._changePointNet(x_nxt)
                    val_outs=val_outs._replace(cp_logits=cp_logits)
                (valBatchAvg,valCpBatchAvg), _ = self._getLoss(val_outs, val_labels,\
                    runAvgObj=valRunAvgObj, cpRunAvgObj=valCpRunAvgObj)
        
                #logging
                if wandb:
                    self._logger(wandb, batchAvg=valBatchAvg, batchCpAvg=valCpBatchAvg, batch_num=i)
                    idx = np.random.choice(val_x.shape[0],10)
                    idxSample, idxLabel = self._getSample(out=val_outs, label=val_labels, vcf_idx=vcf_idx, idx=idx)
                    if plotObj is not None: self._plotSample(plotObj=plotObj, idxSample=idxSample, \
                        idxLabel=idxLabel, idx=idx, phase="valid")
                del val_x, val_y, cps, val_labels, val_outs
    
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=valBatchAvg, t_cp_accr=valCpBatchAvg, t_out=val_outs)

    @timer
    def pred(self, data_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        for _, m in self.model.items():
                m.eval()
        with torch.no_grad():
            for i, data_gen in enumerate(data_generator):
                data_x = data_gen
                data_x = data_x[:, 0:self.params.chmlen].float().to(self.params.device)
                
                if self.params.mc_dropout:
                    activate_mc_dropout(*list(self.model.values()))
                else:
                    assert self.params.mc_samples==1, "MC dropout disabled"

                outs_list, x_nxt_list=[],[]
                for _ in range(self.params.mc_samples):
                    outs_tmp, x_nxt_tmp = self._inner(data_x)
                    outs_list.append(outs_tmp)
                    x_nxt_list.append(x_nxt_tmp)
                    
                outs = self._getFromMcSamples(outs_list)
                x_nxt = self._getFromMcSamples(x_nxt_list)
                if self.params.cp_predict: 
                    cp_logits = self._changePointNet(x_nxt)
                    outs=outs._replace(cp_logits=cp_logits)
                
                #logging
                if wandb:
                    idx = np.random.choice(data_x.shape[0],1)
                    idxSample = self._getSample(out=outs, idx=idx)
                    self._plotSample(idxSample=idxSample)
                    if plotObj is not None: self._plotSample(idxSample=idxSample)
                del data_x, outs_tmp
        # delete tensors for memory optimization
        
        torch.cuda.empty_cache()
        return outs

    def _get_n_comp(self):
        if self.params.residual:
            n_comp = self.params.n_comp_overall 
        else:
            n_comp = self.params.n_comp_overall + self.params.n_comp_subclass
        return n_comp

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
        outs, x_nxt = self._inner(x)
        if self.params.cp_predict: 
            cp_logits = self._changePointNet(x_nxt)
            outs=outs._replace(cp_logits=cp_logits)
        loss, lossBack = self._getLoss(outs, target, mask, runAvgObj=runAvgObj) 
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
            mse_loss= self.MseLosss(y.coord, target.coord).item(),\
            smoothl1_loss= self.smoothL1Loss(y.coord, target.coord, self.params.device),\
            )
        batchCpLoss=batchCpLoss._replace(Precision=precision,\
            Recall=recall,\
            BalancedAccuracy=balanced_accuracy)
        
        # update the running avg object
        for key, val in runAvgObj.items():
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
            Balanced_Accuracy=cpRunAvgObj.get('Balanced_Accuracy'))

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
        y_idx = y.coord[idx,:,:self.params.n_comp_overall].cpu().numpy().reshape(-1, self.params.n_comp_overall)
        if self.params.superpop_predict:
            y_sp_idx = y.sp[idx,:].detach().cpu().numpy().reshape(1,-1)
        vcf_idx = data_vcf_idx[idx,:].detach().cpu().numpy().reshape(-1, 1)
        return y_idx, y_sp_idx, target_idx, vcf_idx

    def _plotSample(self, wandb, plotObj, *kwargs):
        idx=kwargs.get('idx')
        phase=kwargs.get('phase')
        y = kwargs.get('idxSample')
        target = kwargs.get('idxLabel')
        y_args=self._getSample(y, target, idx)
        fig, ax = plotObj.plot_index(*y_args)
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
            