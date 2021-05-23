import torch
import numpy as np
import random
from src.utils.decorators import timer
from src.utils.modelUtil import activate_mc_dropout
from src.utils.dataUtil import getValueBySelection, square_normalize, get_gradient
from src.main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss, \
    gradient_reg, eval_cp_batch, t_accr, t_out, t_cp_accr, t_results, \
        t_balanced_gcd, Running_Average
from src.main.modelSelection import Selections
import pdb
import snoop

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
    @snoop    
    @timer
    def train(self, optimizer, training_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        debugMode = kwargs.get('debugMode')
        # debugMode = True
        trainRunAvgObj = {k:Running_Average() for k in t_accr._fields}
        trainCpRunAvgObj = {k:Running_Average() for k in t_cp_accr._fields}
        trainGcdLoss=GcdLoss() if self.params.geography else None
        for _, m in self.model.items():
            m.train()

        for i, train_gen in enumerate(training_generator):
            train_x, train_y, vcf_idx, cps, superpop , granularpop= train_gen
            train_x = train_x.to(self.params.device)
            train_y = train_y.to(self.params.device)
            cps = cps.unsqueeze(2).to(self.params.device)
            cp_mask = (cps==0) # mask for transition windows
            train_labels = t_out(coord_main=train_y, cp_logits=cps.float()) #BCE needs float as target and not byte 
            
            if debugMode: self._checkModelParamGrads()
            
            # Forward pass
            # update the gradients to zero
            # replace optimizer.zero_grad() with the below
            for _, m in self.model.items():
                for param in m.parameters():
                    param.grad = None

            if debugMode: self._checkModelParamGrads()

            train_outs, loss_outer, lossBack = self._outer(train_x, train_labels, cp_mask)
            lossBack.backward()
            sample_size=cp_mask.sum()

            #check that the model param grads are not None
            self._checkModelParamGrads()

            # update the weights
            optimizer.step()

            if debugMode: self._checkModelParamGrads()

            # evaluate other accuracy for reporting
            trainBatchAvg, trainCpBatchAvg = self._evaluateAccuracy(train_outs, train_labels, \
            sample_size=sample_size, batchLoss=t_accr(loss_main=loss_outer.t_accr.loss_main, loss_aux=loss_outer.t_accr.loss_aux), \
            batchCpLoss=loss_outer.t_cp_accr, runAvgObj=trainRunAvgObj, cpRunAvgObj=trainCpRunAvgObj, gcdObj=trainGcdLoss)

            trainBalancedGcd=None
            if self.params.geography:
                trainBalancedGcd = self.getBalancedClassGcd(trainGcdLoss, superpop, granularpop)
            
            #logging
            if wandb:
                self._logger(wandb, batchAvg=trainBatchAvg, batch_num=i)
                # idx = np.random.choice(train_x.shape[0],1)[0]
                idx=30
                idxSample, idxLabel, idxVcf_idx = self._getSample(out=train_outs, label=train_labels, vcf_idx=vcf_idx, idx=idx)
                if plotObj is not None and random.uniform(0,1)>0.5: self._plotSample(wandb, plotObj, idxSample=idxSample, \
                    idxLabel=idxLabel, idx=idx, idxVcf_idx=idxVcf_idx)
            del train_x, train_y, cps
    
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=trainBatchAvg, t_cp_accr=trainCpBatchAvg, t_out=train_outs, t_balanced_gcd=trainBalancedGcd)
    @snoop
    @timer
    @torch.no_grad()
    def valid(self, validation_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        valRunAvgObj = {k:Running_Average() for k in t_accr._fields}
        valCpRunAvgObj = {k:Running_Average() for k in t_cp_accr._fields}
        valGcdLoss=GcdLoss()
        valPredLs, valVarLs=[],[]

        for _, m in self.model.items():
            m.eval()

        for i, val_gen in enumerate(validation_generator):
            val_x, val_y, vcf_idx, cps, superpop, granularpop = val_gen
            val_x = val_x.to(self.params.device)
            val_y = val_y.to(self.params.device)
            cps = cps.unsqueeze(2).to(self.params.device)
            val_labels = t_out(coord_main=val_y, cp_logits=cps.float()) #BCE needs float as target and not byte 

            if self.params.mc_dropout:
                activate_mc_dropout(*list(self.model.values()))
            else:
                self.params.mc_samples=1
                # assert self.params.mc_samples==1, "MC dropout disabled"

            val_outs_list, x_nxt_list, val_aux_list=[],[],[]
            for _ in range(self.params.mc_samples):
                val_outs, x_nxt = self._inner(val_x, target=val_labels)
                # only collect and mc dropout for the main network
                val_outs_list.append(val_outs.coord_main)
                val_aux_list.append(val_outs.coord_aux)
                x_nxt_list.append(x_nxt)
                
            if self.params.mc_dropout:
                val_outs_main, y_var = self._getFromMcSamples(val_outs_list, getVariance=True)
                val_outs_aux, _ = self._getFromMcSamples(val_aux_list)
                val_outs=t_out(coord_main=val_outs_main, coord_aux=val_outs_aux, y_var=y_var)
                x_nxt, _ = self._getFromMcSamples(x_nxt_list)
                valVarLs.append(val_outs.y_var.detach().cpu().numpy())
            
            valPredLs.append(val_outs.coord_main.detach().cpu().numpy())
            loss_inner=self._getLossInner(val_outs, val_labels)

            cp_accr=None
            if self.params.cp_predict: 
                cp_logits, cp_accr = self._changePointNet(x_nxt, target=val_labels.cp_logits)
                cp_accr=cp_accr._replace(loss_cp=cp_accr.loss_cp.item())
                val_outs=val_outs._replace(cp_logits=cp_logits)                    
            
            sample_size=val_labels.coord_main.shape[0]*val_labels.coord_main.shape[1]
            
            valBatchAvg, valCpBatchAvg = self._evaluateAccuracy(val_outs, val_labels,\
                sample_size=sample_size, batchLoss=t_accr(loss_aux=loss_inner.loss_aux.item(), loss_main=loss_inner.loss_main.item()), \
            batchCpLoss=cp_accr, runAvgObj=valRunAvgObj, cpRunAvgObj=valCpRunAvgObj, gcdObj=valGcdLoss)
            
            valBalancedGcd=None
            if self.params.geography:
                valBalancedGcd = self.getBalancedClassGcd(valGcdLoss, superpop, granularpop)
    
            #logging
            if wandb:
                self._logger(wandb, batchAvg=valBatchAvg, batch_num=i)
                # idx = np.random.choice(val_x.shape[0],1)[0]
                idx=30
                idxSample, idxLabel, idxVcf_idx = self._getSample(out=val_outs, label=val_labels, vcf_idx=vcf_idx, idx=idx)
                if plotObj is not None : self._plotSample(wandb, plotObj, idxSample=idxSample, \
                    idxLabel=idxLabel, idx=idx, idxVcf_idx=idxVcf_idx)
            del val_x, val_y, cps, val_labels
        val_outs=val_outs._replace(coord_main=np.concatenate((valPredLs), axis=0))
        if self.params.mc_dropout:
            val_outs=val_outs._replace(y_var=np.concatenate((valVarLs), axis=0))
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=valBatchAvg, t_cp_accr=valCpBatchAvg, t_out=val_outs, t_balanced_gcd=valBalancedGcd)

    @timer
    def pred(self, data_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        PredLs, VarLs=[], []
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

                outs_list, x_nxt_list, aux_list=[],[], []
                for _ in range(self.params.mc_samples):
                    outs, x_nxt = self._inner(data_x)
                    outs_list.append(outs.coord_main)
                    aux_list.append(outs.coord_aux)
                    x_nxt_list.append(x_nxt)
                    
                if self.params.mc_dropout:
                    outs_main, y_var = self._getFromMcSamples(outs_list, getVariance=True)
                    outs_aux, _ = self._getFromMcSamples(aux_list)
                    x_nxt,_ = self._getFromMcSamples(x_nxt_list)
                    outs=t_out(coord_main=outs_main, coord_aux=outs_aux, y_var=y_var)
                    VarLs.append(outs.y_var.detach().cpu().numpy())
                PredLs.append(outs.coord_main.detach().cpu().numpy())
                if self.params.cp_predict: 
                    cp_logits, _ = self._changePointNet(x_nxt)
                    outs=outs._replace(cp_logits=cp_logits)
                
                #logging
                if wandb:
                    idx = 0
                    idxSample = self._getSample(out=outs, idx=idx)
                    self._plotSample(idxSample=idxSample)
                    if plotObj is not None: self._plotSample(wandb, plotObj, idxSample=idxSample, idx=idx)
                del data_x
            outs=outs._replace(coord_main=np.concatenate((PredLs), axis=0))
            if self.params.mc_dropout:
                outs=outs._replace(y_var=np.concatenate((VarLs), axis=0))
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
            mask=torch.ones_like(target.cp_logits, dtype=torch.uint8)
        out_aux, x_nxt = self._auxNet(x)
        if self.params.geography: out_aux= square_normalize(out_aux)
        outs = t_out(coord_main = out_aux*mask, coord_aux= out_aux*mask)
        return outs, x_nxt

    def _getLossInner(self, outs, target):
        auxLoss=self.criterion(outs.coord_aux, target.coord_main)
        mainLoss=auxLoss
        return t_accr(loss_main=mainLoss, loss_aux=auxLoss)
    
    def _outer(self, x, target, mask):
        outs, x_nxt= self._inner(x, target=target, mask=mask)
        loss_inner=self._getLossInner(outs, target)
        sample_size=mask.sum()
        lossBack=loss_inner.loss_aux/sample_size
        accr = t_results(t_accr(loss_aux=loss_inner.loss_aux.item(), loss_main=loss_inner.loss_main.item()))
        if self.params.cp_predict: 
            cp_logits, cp_accr = self._changePointNet(x_nxt, target=target.cp_logits)
            lossBack+=cp_accr.loss_cp/(target.cp_logits.shape[0]*target.cp_logits.shape[1])
            accr=accr._replace(t_cp_accr=cp_accr._replace(loss_cp=cp_accr.loss_cp.item()))
            outs=outs._replace(cp_logits=cp_logits)
        return outs, accr, lossBack

    def _changePointNet(self, x, **kwargs):
        target=kwargs.get('target')
        cp_logits = self.model['cp'](x)
        loss_cp = self.BCEwithLogits(cp_logits, target)
        cp_pred = (torch.sigmoid(cp_logits)>0.5).int()
        cp_pred=cp_pred.squeeze(2)
        cp_accr=None
        target=target.squeeze(2)
        if target is not None:
            precision, recall, _, _, balanced_accuracy = eval_cp_batch(target, cp_pred, self.params.n_win) if self.params.evalCp else (None,)*5
            cp_accr=t_cp_accr(loss_cp=loss_cp, Precision=precision, Recall=recall, BalancedAccuracy=balanced_accuracy)
        return cp_logits, cp_accr
    
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
            smoothl1_loss= self.smoothL1Loss(y.coord_main, target.coord_main).item(),\
            weighted_loss=self.WeightedLoss(y.coord_main, target.coord_main).item())
        
        if gcdObj is not None:
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
            cpRunAvgObj['loss_cp'].update(batchCpLoss.loss_cp, target.cp_logits.shape[0]*target.cp_logits.shape[1])
            if getattr(batchCpLoss, 'Precision') is not None :cpRunAvgObj['Precision'].update(batchCpLoss.Precision, 1) 
            if getattr(batchCpLoss, 'Recall') is not None :cpRunAvgObj['Recall'].update(batchCpLoss.Recall, 1) 
            if getattr(batchCpLoss, 'BalancedAccuracy') is not None :cpRunAvgObj['BalancedAccuracy'].update(batchCpLoss.BalancedAccuracy, 1) 
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

    def _getFromMcSamples(self, outs_list, getVariance=False):
        cat_outs = torch.cat(outs_list, 0).contiguous()
        mean_outs = cat_outs.view(self.params.mc_samples, -1, self.params.n_win, cat_outs.shape[-1]).mean(0)
        var_outs=None
        if getVariance:
            var_outs = cat_outs.view(self.params.mc_samples, -1, self.params.n_win, cat_outs.shape[-1]).var(0)
        return mean_outs, var_outs

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
        phase="train" if any([m.training for m in list(self.model.values())]) else "valid/test"
        wandb.log({f"MainTask_Loss/{phase}":batchAvg.l1_loss, "batch_num":batch_num})
        wandb.log({f"AuxTask_Loss/{phase}":batchAvg.loss_aux, "batch_num":batch_num})
        
    def _checkModelParamGrads(self):
        for k, m in self.model.items():  
            print("**"*20+f" model {k}"+"**"*20)
            for n, p in m.named_parameters():
                if p.requires_grad:
                    if p.grad is not None: print(n, p.grad) 
                    else: print(f"{n}, None")
