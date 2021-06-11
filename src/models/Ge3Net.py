import torch
import numpy as np
import random
from torch import nn
from src.utils.decorators import timer
from src.utils.modelUtil import activate_mc_dropout
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import GcdLoss, eval_cp_batch, gradient_reg, balancedMetrics, t_results, \
Running_Average, modelOuts, branchLoss, PrCounts, computePrMetric, cpMethod, getCpPred
from src.main.modelSelection import Selections
from src.models.MCDropout import MC_Dropout
from dataclasses import fields
from copy import deepcopy
import pdb
import snoop

class Ge3NetBase():
    def __init__(self, params, model):
        self.params = params
        self.option = Selections.get_selection()
        self.criterion = self.option['loss'][self.params.criteria](reduction='sum', \
            alpha=self.params.criteria_alpha, geography=self.params.geography)
        self.losses = {metric: self.option['loss'][metric](reduction='sum', \
            alpha=self.params.criteria_alpha, geography=self.params.geography) \
                for metric in self.option['loss'] if metric!=self.params.criteria}
        self.model=model

    def getRunningAvgObj(self):
        lossesLs = list(self.losses.keys())
        branchLosses = [field.name for field in fields(branchLoss)]
        if self.params.evalExtraMainLosses: lossesLs +=branchLosses[0:2]
        runAvgObj={metric:Running_Average() for metric in lossesLs}
        cpRunAvgObj=None
        if self.params.cp_predict:
            cpMetricLs=[branchLosses[-1]]
            cpRunAvgObj={metric:Running_Average() for metric in cpMetricLs}
            if self.params.evalCp:cpRunAvgObj["prCounts"]=PrCounts()
        return runAvgObj, cpRunAvgObj

    @timer
    def batch_train(self, optimizer, training_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        debugMode = kwargs.get('debugMode')
        trainRunAvgObj, trainCpRunAvgObj = self.getRunningAvgObj()
        trainGcdBalancedMetricsObj=balancedMetrics() if self.params.geography else None
        
        self.model.train()

        for i, train_gen in enumerate(training_generator):
            train_x, train_y, vcf_idx, cps, superpop , granularpop= train_gen
            train_x = train_x.to(self.params.device)
            train_y = train_y.to(self.params.device)
            cps = cps.unsqueeze(2).to(self.params.device)
            cp_mask = (cps==0) # mask for transition windows
            train_labels = modelOuts(coord_main=train_y, cp_logits=cps.float()) #BCE needs float as target and not as bytes 
            
            # Forward pass
            # update the gradients to zero # replace optimizer.zero_grad() with the below
            for param in self.model.parameters():
                param.grad = None

            if debugMode: self.model._checkModelParamGrads()

            train_outs, loss_inner, lossBack = self.model._batch_train_1_step(train_x, train_labels, cp_mask, self.criterion)
            lossBack.backward()

            #check that the model param grads are not None
            if debugMode: self.model._checkModelParamGrads()

            # update the weights
            optimizer.step()

            sample_size=cp_mask.sum()
            # evaluate other accuracy for reporting
            trainBatchAvg, trainCpBatchAvg, trainBalancedGcd = self._evaluate(train_outs, train_labels, \
                sample_size, cp_mask, superpop, granularpop, trainRunAvgObj, trainCpRunAvgObj, \
                    trainGcdBalancedMetricsObj, loss_inner)
            
            #logging
            if wandb:
                self._logOutput(wandb, plotObj, batchAvg=trainBatchAvg, batch_num=i, vcf_idx=vcf_idx, \
                    out=train_outs.coord_main, label=train_labels.coord_main)
            del train_x, train_y, cps

        trainCpBatchAvg['prMetrics']=computePrMetric(trainCpRunAvgObj['prCounts'])
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=trainBatchAvg, t_cp_accr=trainCpBatchAvg, t_out=train_outs, t_balanced_gcd=trainBalancedGcd)

    @timer
    @torch.no_grad()
    def batch_valid(self, validation_generator, **kwargs):
        wandb = kwargs.get('wandb')
        plotObj = kwargs.get('plotObj')
        valRunAvgObj, valCpRunAvgObj = self.getRunningAvgObj()
        valGcdBalancedMetricsObj=balancedMetrics() if self.params.geography else None
        valPredLs, valVarLs, valCpLs=[],[],[]

        self.model.eval()
        mc_dropout = MC_Dropout(self.params.mc_samples, variance=True) if self.params.mc_dropout else None

        for i, val_gen in enumerate(validation_generator):
            val_x, val_y, vcf_idx, cps, superpop, granularpop = val_gen
            val_x = val_x.to(self.params.device)
            val_y = val_y.to(self.params.device)
            cps = cps.unsqueeze(2).to(self.params.device)
            cp_mask=(cps==0)
            val_labels = modelOuts(coord_main=val_y, cp_logits=cps.float()) #BCE needs float as target and not byte 

            val_outs, val_outs_list, loss_inner = self.model._batch_validate_1_step(val_x, val_labels, cp_mask, mc_dropout=mc_dropout)

            if self.params.rtnOuts:   
                valPredLs.append(torch.stack(val_outs_list, dim=0).contiguous().detach().cpu().numpy())
                if self.params.cp_predict:valCpLs.append(val_outs.cp_logits.detach().cpu().numpy()) 
                if self.params.mc_dropout:valVarLs.append(val_outs.y_var.detach().cpu().numpy())                  
            
            batchSize=val_labels.coord_main.shape[0]
            seqLen=val_labels.coord_main.shape[1]
            sample_size=batchSize*seqLen
            # evaluate other accuracy for reporting
            valBatchAvg, valCpBatchAvg, valBalancedGcd = self._evaluate(val_outs, val_labels, \
                sample_size, cp_mask, superpop, granularpop, valRunAvgObj, valCpRunAvgObj, \
                    valGcdBalancedMetricsObj, loss_inner)
    
            #logging
            if wandb:
                self._logOutput(wandb, plotObj, batchAvg=valBatchAvg, batch_num=i, vcf_idx=vcf_idx, \
                    out=val_outs.coord_main, label=val_labels.coord_main)

            del val_x, val_y, cps, val_labels

        if self.params.rtnOuts:
            val_outs.coord_main=np.concatenate((valPredLs), axis=1)
            if self.params.cp_predict:val_outs.cp_logits=np.concatenate((valCpLs), axis=0)
            if self.params.mc_dropout: val_outs.y_var=np.concatenate((valVarLs), axis=0)
        
        valCpBatchAvg['prMetrics']=computePrMetric(valCpRunAvgObj['prCounts'])
        # delete tensors for memory optimization
        torch.cuda.empty_cache()
        return t_results(t_accr=valBatchAvg, t_cp_accr=valCpBatchAvg, t_out=val_outs, t_balanced_gcd=valBalancedGcd)

    def _evaluate(self, outs, labels, sample_size, mask, superpop, granularpop, \
        runAvgObj, cpRunAvgObj, gcdBalancedMetricsObj, loss_inner):
        BatchAvg, CpBatchAvg = self._evaluateAccuracy(outs, labels, \
            sample_size=sample_size, batchLoss={'loss_main':loss_inner.loss_main, \
                'loss_aux':loss_inner.loss_aux}, batchCpLoss={'loss_cp':loss_inner.loss_cp}, \
                    runAvgObj=runAvgObj, cpRunAvgObj=cpRunAvgObj)

        BalancedGcd=None
        if self.params.geography and self.params.evalBalancedGcd:
            gcdMatrix=GcdLoss().rawGcd(outs.coord_main*mask, labels.coord_main*mask).detach()
            BalancedGcd=self.getExtraGcdMetrics(gcdBalancedMetricsObj, gcdMatrix, superpop, granularpop)

        return BatchAvg, CpBatchAvg, BalancedGcd

    # Need to think this one
    def _evaluateAccuracy(self, y, target, **kwargs):
        runAvgObj=kwargs.get('runAvgObj')
        cpRunAvgObj=kwargs.get('cpRunAvgObj')
        batchLoss=kwargs.get('batchLoss')
        batchCpLoss=kwargs.get('batchCpLoss')
        sample_size=kwargs.get('sample_size')
        cpThresh=kwargs.get('cpThresh')
        mask = kwargs.get('mask')
        if mask is None: mask = 1.0
        if self.params.evalExtraMainLosses:
            batchLoss={**batchLoss, **{metric:self.losses[metric](y.coord_main*mask, target.coord_main*mask).item() for metric in self.losses if self.losses[metric] is not None}}
            
        # update the running avg object
        for key, val in runAvgObj.items():
            if batchLoss.get(key) is not None:
                val.update(batchLoss[key], sample_size)
        
        # get the running average for batches in this epoch so far by calling the 
        # running avg object       
        batchAvg={metric:runAvgObj.get(metric)() if batchLoss.get(metric) is not None else None for metric in runAvgObj}
        del batchLoss

        batchCpAvg=None
        if self.params.cp_predict:  
            if cpThresh is None: cpThresh=0.45
            cp_pred = (torch.sigmoid(y.cp_logits)>cpThresh).int()
            cp_pred=cp_pred.squeeze(2) 
            if self.params.evalCp:
                prCounts= eval_cp_batch(target.cp_logits.squeeze(2), cp_pred, self.params.n_win)
                cpRunAvgObj["prCounts"].update(prCounts)
            # update the running avg object
            numSamples=target.cp_logits.shape[0]
            numWin=target.cp_logits.shape[1]
            cpRunAvgObj["loss_cp"].update(batchCpLoss["loss_cp"], numSamples*numWin)
            batchCpAvg={"loss_cp":cpRunAvgObj["loss_cp"]()}
            
            del batchCpLoss
        torch.cuda.empty_cache()
        return batchAvg, batchCpAvg

    def getBalancedClassGcd(self, superpop, granularpop, gcdObj):
        balancedGcdMetrics={}
        gcdObj.balancedMetric(superpop, granularpop)
        meanBalancedGcd=self.option['balancedMetrics']['meanBalanced'](gcdObj)()
        medianBalancedGcd=self.option['balancedMetrics']['medianBalanced'](gcdObj)()
        balancedGcdMetrics['meanBalancedGcdSp'], balancedGcdMetrics['meanBalancedGcdGp'] = meanBalancedGcd
        balancedGcdMetrics['medianBalancedGcdSp'], balancedGcdMetrics['medianBalancedGcdGp'] = medianBalancedGcd
        balancedGcdMetrics['median']=self.option['balancedMetrics']['median'](gcdObj)()
        return  balancedGcdMetrics

    def getExtraGcdMetrics(self, gcdMetricsObj, gcdMatrix, superpop, granularpop):        
        gcdMetricsObj.fillData(gcdMatrix.clone())
        trainBalancedGcd = self.getBalancedClassGcd(superpop, granularpop, gcdMetricsObj)
        accAtGcd=gcdMetricsObj.accAtThresh()
        return trainBalancedGcd

    def _getSample(self,**kwargs):
        idx=kwargs.get('idx')
        data_vcf_idx=kwargs.get('vcf_idx')
        target=kwargs.get('label')
        y=kwargs.get('out')
        target_idx = target[idx,...].detach().cpu().numpy().reshape(-1, self.params.dataset_dim)
        y_idx = y[idx,:,:self.params.n_comp_overall].detach().cpu().numpy().reshape(-1, self.params.n_comp_overall)
        vcf_idx = data_vcf_idx[idx,:].detach().cpu().numpy().reshape(-1, 1)
        return y_idx, target_idx, vcf_idx

    def _plotSample(self, wandb, plotObj, **kwargs):
        idx=kwargs.get('idx')
        idxSample = kwargs.get('idxSample')
        idxLabel = kwargs.get('idxLabel')
        vcf_idx= kwargs.get('idxVcf_idx')
        fig, fig_geo = plotObj.plot_index(idxSample, idxLabel, vcf_idx)
        phase = kwargs.get('phase')
        # phase="train" if any([m.training for m in list(self.model.values())]) else "valid/test"
        wandb.log({f" Image for idx {idx} for {phase}":wandb.Image(fig)})
        if self.params.geography:
            wandb.log({f" Image for idx {idx} for {phase} geography":fig_geo})

    def _logger(self, wandb, **kwargs):
        batch_num=kwargs.get('batch_num')
        batchAvg=kwargs.get('batchAvg')
        phase = kwargs.get('phase')
        # phase="train" if any([m.training for m in list(self.model.values())]) else "valid/test"
        wandb.log({f"MainTask_Loss/{phase}":batchAvg["loss_main"], "batch_num":batch_num})
        wandb.log({f"AuxTask_Loss/{phase}":batchAvg["loss_aux"], "batch_num":batch_num})

    def _checkModelParamGrad(self):
        print("**"*20+f" model {self.model.__class__.__name__}"+"**"*20)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None: print(n, p.grad) 
                else: print(f"{n}, None")

    def _logOutput(self, wandb, plotObj, batchAvg, batch_num, vcf_idx, out, label):
        self._logger(wandb, batchAvg=batchAvg, batch_num=batch_num)
        if plotObj is not None : 
            # idx = np.random.choice(val_x.shape[0],1)[0]
            idx=30
            idxSample, idxLabel, idxVcf_idx = self._getSample(out=out, label=label \
                , vcf_idx=vcf_idx, idx=idx)
            self._plotSample(wandb, plotObj, idxSample=idxSample, idxLabel=idxLabel \
                , idx=idx, idxVcf_idx=idxVcf_idx)


