import torch
import numpy as np
import random
from src.utils.decorators import timer
from src.utils.modelUtil import activate_mc_dropout
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss, \
    gradient_reg, eval_cp_batch, t_accr, t_out, t_cp_accr, t_results, \
        t_balanced_gcd, Running_Average
from src.main.modelSelection import Selections

class model_A(object):
    _network=['aux', 'cp']
    def __init__(self, *args, params):
        self.model={}
        for k,m in zip(self._network, args):
            self.model[k]=m
        self.params = params
        self._init_loss()
        
    def _init_loss(self):
        self.selectLoss=Selections.get_selection()
        self.criterion = self.selectLoss['loss'][self.params.criteria](reduction='sum', alpha=self.params.criteria_alpha)
        self.smoothL1Loss = SmoothL1Loss(reduction='sum', beta=self.params.SmoothLoss_beta)
        self.L1Loss = torch.nn.L1Loss(reduction='sum')
        self.MseLoss = torch.nn.MSELoss(reduction='sum')
        self.BCEwithLogits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.WeightedLoss=Weighted_Loss(reduction='sum', alpha=self.params.weightLoss_alpha)
        
    @timer
    def train(self, optimizer, training_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        trainRunAvgObj = {k:Running_Average() for k in t_accr._fields}
        trainCpRunAvgObj = {k:Running_Average() for k in t_cp_accr._fields}
        trainGcdLoss=GcdLoss()
        
        for _, m in self.model.items():
            m.train()

        for i, train_gen in enumerate(training_generator):
            train_x, train_y, vcf_idx, cps, superpop , granularpop= train_gen
            train_x = train_x.to(self.params.device)
            train_y = train_y.to(self.params.device)
            cps = cps.to(self.params.device)
            cp_mask = (cps==0).float() # mask for transition windows
            train_labels = t_out(coord_main=train_y, cp_logits=cps[:,:,0].unsqueeze(2))
            
            # Forward pass
            # update the gradients to zero
            optimizer.zero_grad()

            train_outs, loss_outer = self._outer(train_x, train_labels, cp_mask)
            sample_size=cp_mask[...,0].sum()

            # update the weights
            optimizer.step()

            # evaluate other accuracy for reporting
            trainBatchAvg, trainCpBatchAvg = self._evaluateAccuracy(train_outs, train_labels, \
            sample_size=sample_size, batchLoss=t_accr(loss_main=loss_outer.t_accr.loss_main, loss_aux=loss_outer.t_accr.loss_aux), \
            batchCpLoss=loss_outer.t_cp_accr, runAvgObj=trainRunAvgObj, cpRunAvgObj=trainCpRunAvgObj, gcdObj=trainGcdLoss)

            trainBalancedGcd=None
            if self.params.geography:
                trainBalancedGcd = self.getBalancedClassGcd(trainGcdLoss, superpop, granularpop)
            
            #logging
            if wandb:
                self._logger(wandb, batchAvg=trainBatchAvg, batchCpAvg=trainCpBatchAvg, batch_num=i, balancedGcd=trainBalancedGcd)
                # idx = np.random.choice(train_x.shape[0],1)[0]
                idx=30
                idxSample, idxLabel, idxVcf_idx = self._getSample(out=train_outs, label=train_labels, vcf_idx=vcf_idx, idx=idx)
                if plotObj is not None and random.uniform(0,1)>0.5: self._plotSample(wandb, plotObj, idxSample=idxSample, \
                    idxLabel=idxLabel, idx=idx, idxVcf_idx=idxVcf_idx)
            del train_x, train_y, cps
    
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=trainBatchAvg, t_cp_accr=trainCpBatchAvg, t_out=train_outs, t_balanced_gcd=trainBalancedGcd)

    @timer
    def valid(self, validation_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        valRunAvgObj = {k:Running_Average() for k in t_accr._fields}
        valCpRunAvgObj = {k:Running_Average() for k in t_cp_accr._fields}
        valGcdLoss=GcdLoss()
        valPredLs, valVarLs=[],[]

        for _, m in self.model.items():
            m.eval()

        with torch.no_grad():
            for i, val_gen in enumerate(validation_generator):
                val_x, val_y, vcf_idx, cps, superpop, granularpop = val_gen
                val_x = val_x.to(self.params.device)
                val_y = val_y.to(self.params.device)
                cps = cps.to(self.params.device)
                val_labels = t_out(coord_main=val_y, cp_logits=cps[:,:,0].unsqueeze(2))

                if self.params.mc_dropout:
                    activate_mc_dropout(*list(self.model.values()))
                else:
                    self.params.mc_samples=1
                    # assert self.params.mc_samples==1, "MC dropout disabled"

                val_outs_list, x_nxt_list=[],[]
                for _ in range(self.params.mc_samples):
                    val_outs_tmp, x_nxt_tmp, _ = self._inner(val_x, target=val_labels)
                    # only collect and mc dropout for the main network
                    val_outs_list.append(val_outs_tmp.coord_main)
                    x_nxt_list.append(x_nxt_tmp)
                    
                val_outs = self._getFromMcSamples(val_outs_list)
                x_nxt = self._getFromMcSamples(x_nxt_list)
                valPredLs.append(val_outs.coord_main.detach().cpu().numpy())
                valVarLs.append(val_outs.y_var.detach().cpu().numpy())
                loss_inner=self._getLossInner(val_outs, val_labels)

                cp_accr=None
                if self.params.cp_predict: 
                    cp_logits, cp_accr = self._changePointNet(x_nxt.coord_main, target=val_labels.cp_logits)
                    val_outs=val_outs._replace(cp_logits=cp_logits)                    
                
                sample_size=val_labels.coord_main.shape[0]*val_labels.coord_main.shape[1]
                
                valBatchAvg, valCpBatchAvg = self._evaluateAccuracy(val_outs, val_labels,\
                    sample_size=sample_size, batchLoss=t_accr(loss_aux=loss_inner.loss_aux, loss_main=loss_inner.loss_main), \
                batchCpLoss=cp_accr, runAvgObj=valRunAvgObj, cpRunAvgObj=valCpRunAvgObj, gcdObj=valGcdLoss)
                
                valBalancedGcd=None
                if self.params.geography:
                    valBalancedGcd = self.getBalancedClassGcd(valGcdLoss, superpop, granularpop)
        
                #logging
                if wandb:
                    self._logger(wandb, batchAvg=valBatchAvg, batchCpAvg=valCpBatchAvg, batch_num=i, balancedGcd=valBalancedGcd)
                    # idx = np.random.choice(val_x.shape[0],1)[0]
                    idx=30
                    idxSample, idxLabel, idxVcf_idx = self._getSample(out=val_outs, label=val_labels, vcf_idx=vcf_idx, idx=idx)
                    if plotObj is not None : self._plotSample(wandb, plotObj, idxSample=idxSample, \
                        idxLabel=idxLabel, idx=idx, idxVcf_idx=idxVcf_idx)
                del val_x, val_y, cps, val_labels
        val_outs=val_outs._replace(coord_main=np.concatenate((valPredLs), axis=0), y_var=np.concatenate((valVarLs), axis=0))
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=valBatchAvg, t_cp_accr=valCpBatchAvg, t_out=val_outs, t_balanced_gcd=valBalancedGcd)

    @timer
    def pred(self, data_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        for _, m in self.model.items():
                m.eval()
        with torch.no_grad():
            for i, data_gen in enumerate(data_generator):
                data_x = data_gen
                data_x = data_x.to(self.params.device)
                
                if self.params.mc_dropout:
                    activate_mc_dropout(*list(self.model.values()))
                else:
                    self.params.mc_samples=1

                outs_list, x_nxt_list=[],[]
                for _ in range(self.params.mc_samples):
                    outs_tmp, x_nxt_tmp, _ = self._inner(data_x)
                    outs_list.append(outs_tmp)
                    x_nxt_list.append(x_nxt_tmp)
                    
                outs = self._getFromMcSamples(outs_list)
                x_nxt = self._getFromMcSamples(x_nxt_list)
                if self.params.cp_predict: 
                    cp_logits, cp_accr = self._changePointNet(x_nxt.coord)
                    outs=outs._replace(cp_logits=cp_logits)
                
                #logging
                if wandb:
                    idx = 0
                    idxSample = self._getSample(out=outs, idx=idx)
                    self._plotSample(idxSample=idxSample)
                    if plotObj is not None: self._plotSample(wandb, plotObj, idxSample=idxSample, idx=idx)
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

    def _inner(self, x, **kwargs):
        target=kwargs.get('target')
        mask = kwargs.get('mask')
        if mask is None and target is not None:
            mask=torch.ones_like(target.coord_main, dtype=float)
        out_aux, x_nxt = self._auxNet(x)
        if self.params.geography: out_aux= square_normalize(out_aux)
        outs = t_out(coord_main = out_aux*mask)
        if target is not None:
            loss_aux = self.criterion(out_aux*mask, target.coord_main*mask)
            loss_inner = t_accr(loss_aux=loss_aux, loss_main=loss_aux)
        return outs, x_nxt, loss_inner

    def _getLossInner(self, x, target):
        loss=self.criterion(x.coord_main, target.coord_main)
        return t_accr(loss_aux=loss, loss_main=loss)

    def _outer(self, x, target, mask):
        outs, x_nxt, loss_inner= self._inner(x, target=target, mask=mask)
        sample_size=mask[...,0].sum()
        lossBack=loss_inner.loss_aux/sample_size
        
        if self.params.cp_predict: 
            cp_logits, cp_accr = self._changePointNet(x_nxt, target=target.cp_logits)
            lossBack+=cp_accr.loss_cp/(target.cp_logits.shape[0]*target.cp_logits.shape[1])
            outs=outs._replace(cp_logits=cp_logits)
        lossBack.backward()
        return outs, t_results(t_accr=t_accr(loss_aux=loss_inner.loss_aux.item(), loss_main=loss_inner.loss_main.item()),\
            t_cp_accr=t_cp_accr(loss_cp=cp_accr.loss_cp.item(), Precision=cp_accr.Precision, Recall=cp_accr.Recall, BalancedAccuracy=cp_accr.BalancedAccuracy))

    def _changePointNet(self, x, **kwargs):
        target=kwargs.get('target')
        cp_logits = self.model['cp'](x)
        loss_cp = self.BCEwithLogits(cp_logits, target)
        cp_pred = (torch.sigmoid(cp_logits)>0.5).int()
        cp_pred=cp_pred.squeeze(2)
        target=target.squeeze(2)
        precision, recall, _, _, balanced_accuracy = eval_cp_batch(target, cp_pred)
        return cp_logits, t_cp_accr(loss_cp=loss_cp, Precision=precision, Recall=recall, BalancedAccuracy=balanced_accuracy)
    
    def _evaluateAccuracy(self, y, target, **kwargs):
        runAvgObj=kwargs.get('runAvgObj')
        cpRunAvgObj=kwargs.get('cpRunAvgObj')
        batchLoss=kwargs.get('batchLoss')
        batchCpLoss=kwargs.get('batchCpLoss')
        sample_size=kwargs.get('sample_size')
        gcdObj=kwargs.get('gcdObj')
        gcdThresh=kwargs.get('gcdThresh')

        batchLoss=batchLoss._replace(l1_loss=self.L1Loss(y.coord_main, target.coord_main).item(),\
            mse_loss= self.MseLoss(y.coord_main, target.coord_main).item(),\
            smoothl1_loss= self.smoothL1Loss(y.coord_main, target.coord_main, self.params.device).item(),\
            weighted_loss=self.WeightedLoss(y.coord_main, target.coord_main).item())
        
        if self.params.geography:
            if gcdThresh is None: gcdThresh=1000.0
            batchLoss=batchLoss._replace(gcdLoss=gcdObj(y.coord_main, target.coord_main).item(),
            accAtGcd=gcdObj.accAtGcd(y.coord_main, target.coord_main, gcdThresh))
            
        # update the running avg object
        for key, val in runAvgObj.items():
            if getattr(batchLoss,key) is not None:
                val.update(getattr(batchLoss,key), sample_size)
        
        # get the running average for batches in this epoch so far by calling the 
        # running avg object       
        batchAvg=t_accr(l1_loss=runAvgObj.get('l1_loss')() if getattr(batchLoss,'l1_loss') is not None else None, \
            loss_aux=runAvgObj.get('loss_aux')() if getattr(batchLoss,'loss_aux') is not None else None, \
            loss_main=runAvgObj.get('loss_main')() if getattr(batchLoss,'loss_main') is not None else None,\
            mse_loss=runAvgObj.get('mse_loss')() if getattr(batchLoss,'mse_loss') is not None else None, \
            smoothl1_loss=runAvgObj.get('smoothl1_loss')() if getattr(batchLoss,'smoothl1_loss') is not None else None, \
            weighted_loss=runAvgObj.get('weighted_loss')() if getattr(batchLoss,'weighted_loss') is not None else None,\
            gcdLoss=runAvgObj.get('gcdLoss')() if getattr(batchLoss,'gcdLoss') is not None else None,\
            accAtGcd=runAvgObj.get('accAtGcd')() if getattr(batchLoss,'accAtGcd') is not None else None  )
        
        batchCpAvg=None
        if self.params.cp_predict:            
            cpRunAvgObj['loss_cp'].update(batchCpLoss.loss_cp.item(), target.cp_logits.shape[0]*target.cp_logits.shape[1])
            cpRunAvgObj['Precision'].update(batchCpLoss.Precision, 1)
            cpRunAvgObj['Recall'].update(batchCpLoss.Recall, 1)
            cpRunAvgObj['BalancedAccuracy'].update(batchCpLoss.BalancedAccuracy, 1)
            batchCpAvg=t_cp_accr(loss_cp=cpRunAvgObj.get('loss_cp')() if getattr(batchCpLoss,'loss_cp') is not None else None,\
            Precision=cpRunAvgObj.get('Precision')() if getattr(batchCpLoss,'Precision') is not None else None,\
            Recall=cpRunAvgObj.get('Recall')() if getattr(batchCpLoss,'Recall') is not None else None,\
            BalancedAccuracy=cpRunAvgObj.get('BalancedAccuracy')() if getattr(batchCpLoss,'BalancedAccuracy') is not None else None)
            del batchCpLoss

        del batchLoss

        torch.cuda.empty_cache()
        return batchAvg, batchCpAvg

    def getBalancedClassGcd(self, gcdObj, superpop, granularpop):
        gcdObj.balancedGcd(superpop, granularpop)
        runAvgMedianGcd=gcdObj.median()
        meanBalancedGcd=gcdObj.meanBalanced()
        medianBalancedGcd=gcdObj.medianBalanced()
        return t_balanced_gcd(median_gcd=runAvgMedianGcd, meanBalancedGcdSp=meanBalancedGcd[0], meanBalancedGcdGp=meanBalancedGcd[1],\
            medianBalancedGcdSp=medianBalancedGcd[0], medianBalancedGcdGp=medianBalancedGcd[1])

    def _getFromMcSamples(self, outs_list):
        cat_outs = torch.cat(outs_list, 0).contiguous()
        mean_outs = cat_outs.view(self.params.mc_samples, -1, self.params.n_win, cat_outs.shape[-1]).mean(0)
        var_outs = cat_outs.view(self.params.mc_samples, -1, self.params.n_win, cat_outs.shape[-1]).var(0)
        return t_out(coord_main=mean_outs, y_var=var_outs)

    def _getSample(self,**kwargs):
        idx=kwargs.get('idx')
        data_vcf_idx=kwargs.get('vcf_idx')
        target=kwargs.get('label')
        y=kwargs.get('out')
        target_idx = target.coord_main[idx,...].detach().cpu().numpy().reshape(-1, self.params.dataset_dim)
        y_idx = y.coord_main[idx,:,:self.params.n_comp_overall].detach().cpu().numpy().reshape(-1, self.params.n_comp_overall)
        if self.params.superpop_predict:
            y_sp_idx = y.sp[idx,:].detach().cpu().numpy().reshape(1,-1)
        vcf_idx = data_vcf_idx[idx,:].detach().cpu().numpy().reshape(-1, 1)
        return y_idx, target_idx, vcf_idx

    def _plotSample(self, wandb, plotObj, **kwargs):
        idx=kwargs.get('idx')
        idxSample = kwargs.get('idxSample')
        idxLabel = kwargs.get('idxLabel')
        vcf_idx= kwargs.get('idxVcf_idx')
        fig, fig_geo = plotObj.plot_index(idxSample, idxLabel, vcf_idx)
        phase="train" if any([m.training for m in list(self.model.values())]) else "valid/test"
        wandb.log({f" Image for idx {idx} for {phase}":wandb.Image(fig)})
        if self.params.geography:
            wandb.log({f" Image for idx {idx} for {phase} geography":fig_geo})

    def _logger(self, wandb, **kwargs):
        batch_num=kwargs.get('batch_num')
        batchAvg=kwargs.get('batchAvg')
        batchCpAvg=kwargs.get('batchCpAvg')
        balancedGcd=kwargs.get('balancedGcd')
        phase="train" if any([m.training for m in list(self.model.values())]) else "valid/test"
        wandb.log({f"MainTask_Loss/{phase}":batchAvg.l1_loss, "batch_num":batch_num})
        wandb.log({f"AuxTask_Loss/{phase}":batchAvg.loss_aux, "batch_num":batch_num})
        # if self.params.geography:
        #     wandb.log({f"more gcd/{phase}":balancedGcd._asdict(), "batch_num":batch_num})
        #     wandb.log({f"more gcd/{phase}_accAtGcd":batchAvg.accAtGcd, "batch_num":batch_num})
        # if self.params.cp_predict:
        #     wandb.log({f"loss_cp/{phase}":batchCpAvg.loss_cp,"batch_num":batch_num})
        #     wandb.log({f"cp_accr/{phase}":batchCpAvg._asdict(), "batch_num":batch_num})
        # if self.params.residual:
        #     wandb.log({f"residual_loss/{phase}":batchAvg.residual_loss, "batch_num":batch_num})
            