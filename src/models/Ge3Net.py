import torch
import numpy as np
from torch import nn
import os.path as osp
from src.utils.decorators import timer
from src.utils.modelUtil import save_checkpoint, early_stopping, custom_opt
from src.main.evaluation import GcdLoss, eval_cp_batch, balancedMetrics, t_results, \
Running_Average, modelOuts, branchLoss, PrCounts, computePrMetric
from src.models.MCDropout import MC_Dropout
from dataclasses import fields
import matplotlib.pyplot as plt
import optuna
import pdb
import snoop

class Ge3NetBase():
    def __init__(self, params, model, option):
        self.params = params
        self.option = option
        self.losses = {metric: self.option['loss'][metric](reduction='sum', \
            alpha=self.params.criteria_alpha, geography=self.params.geography) \
                for metric in self.option['loss'] if metric!=self.params.criteria}
        self.model=model
        self.wandb=None
        self.plotObj=None

    def getRunningAvgObj(self):
        lossesLs = list(self.losses.keys())
        branchLosses = [field.name for field in fields(branchLoss)][0:2]
        if self.params.evalExtraMainLosses: branchLosses +=lossesLs
        runAvgObj={metric:Running_Average() for metric in branchLosses}
        cpRunAvgObj=None
        if self.params.cp_predict:
            cpMetricLs=["loss_cp"]
            cpRunAvgObj={metric:Running_Average() for metric in cpMetricLs}
            if self.params.evalCp:cpRunAvgObj["prCounts"]=PrCounts()
        return runAvgObj, cpRunAvgObj

    @timer
    def batchLoopTrain(self, optimizer, training_generator, **kwargs):
        debugMode = kwargs.get('debugMode')
        trainRunAvgObj, trainCpRunAvgObj = self.getRunningAvgObj()
        trainGcdBalancedMetricsObj=balancedMetrics() if self.params.geography else None
        self.model.train()

        for i, train_gen in enumerate(training_generator):
            train_x, train_y, vcf_idx, cps, superpop , granularpop= train_gen
            train_x = train_x.to(self.params.device).float()
            train_y = train_y.to(self.params.device).float()
            cps = cps.to(self.params.device).unsqueeze(2)
            cp_mask = (cps==0) # mask for transition windows
            train_labels = modelOuts(coord_main=train_y, cp_logits=cps.float()) #BCE needs float as target and not as bytes 
            
            # Forward pass
            # update the gradients to zero # replace optimizer.zero_grad() with the below
            for param in self.model.parameters():
                param.grad = None

            if debugMode: self.model._checkModelParamGrads()

            train_outs, loss_inner, lossBack = self.model._batch_train_1_step(train_x, train_labels, cp_mask)
            if lossBack is not None: lossBack.backward() 

            #check that the model param grads are not None
            if debugMode: self.model._checkModelParamGrads()

            # update the weights
            optimizer.step()

            # evaluate other accuracy for reporting
            trainBatchAvg, trainCpBatchAvg, trainBalancedGcd = self._evaluate(train_outs, train_labels, \
            cp_mask, superpop, granularpop, trainRunAvgObj, trainCpRunAvgObj, \
            trainGcdBalancedMetricsObj, loss_inner)
            
            #logging
            if self.wandb is not None:
                self._logOutput(batchAvg=trainBatchAvg, batch_num=i, vcf_idx=vcf_idx, \
                    out=train_outs.coord_main, label=train_labels.coord_main)
            del train_x, train_y, cps

        if self.params.evalCp: trainCpBatchAvg['prMetrics']=computePrMetric(trainCpRunAvgObj['prCounts'])
        # delete tensors for memory optimization
        del trainRunAvgObj, trainCpRunAvgObj, trainGcdBalancedMetricsObj
        torch.cuda.empty_cache()
        return t_results(t_accr=trainBatchAvg, t_cp_accr=trainCpBatchAvg, t_out=train_outs, t_balanced_gcd=trainBalancedGcd)

    @timer
    @torch.no_grad()
    def batchLoopValid(self, validation_generator):
        valRunAvgObj, valCpRunAvgObj = self.getRunningAvgObj()
        valGcdBalancedMetricsObj=balancedMetrics() if self.params.geography else None
        valPredLs, valVarLs, valCpLs=[],[],[]
        self.model.eval()
        mc_dropout = MC_Dropout(self.params.mc_samples, self.params.n_win, variance=True) if self.params.mc_dropout else None
        for i, val_gen in enumerate(validation_generator):
            val_x, val_y, vcf_idx, cps, superpop, granularpop = val_gen
            val_x = val_x.to(self.params.device).float()
            val_y = val_y.to(self.params.device).float()
            cps = cps.to(self.params.device).unsqueeze(2)
            cp_mask=(cps==0) if self.params.validCpMask else torch.ones_like(cps, device=self.params.device)
            val_labels = modelOuts(coord_main=val_y, cp_logits=cps.float()) #BCE needs float as target and not byte 

            val_outs, loss_inner = self.model._batch_validate_1_step(val_x, val_labels=val_labels, mc_dropout=mc_dropout, cp_mask=cp_mask)

            if self.params.rtnOuts:   
                valPredLs.append(torch.stack(val_outs.coord_mainLs, dim=0).contiguous().detach().cpu().numpy())
                if self.params.cp_predict:valCpLs.append(val_outs.cp_logits.detach().cpu().numpy()) 
                if self.params.mc_dropout:valVarLs.append(val_outs.y_var.detach().cpu().numpy())                  
            
            # evaluate other accuracy for reporting
            valBatchAvg, valCpBatchAvg, valBalancedGcd = self._evaluate(val_outs, val_labels, \
            cp_mask, superpop, granularpop, valRunAvgObj, valCpRunAvgObj, \
            valGcdBalancedMetricsObj, loss_inner)
            
            #logging
            if self.wandb is not None:
                self._logOutput(batchAvg=valBatchAvg, batch_num=i, vcf_idx=vcf_idx, \
                out=val_outs.coord_main, label=val_labels.coord_main)

            del val_x, val_y, cps, val_labels

        if self.params.rtnOuts:
            val_outs.coord_main=np.concatenate((valPredLs), axis=1)
            if self.params.cp_predict:val_outs.cp_logits=np.concatenate((valCpLs), axis=0)
            if self.params.mc_dropout: val_outs.y_var=np.concatenate((valVarLs), axis=0)
        
        if self.params.evalCp: valCpBatchAvg['prMetrics']=computePrMetric(valCpRunAvgObj['prCounts'])
        # delete tensors for memory optimization
        del valRunAvgObj, valCpRunAvgObj, valGcdBalancedMetricsObj
        torch.cuda.empty_cache()
        return t_results(t_accr=valBatchAvg, t_cp_accr=valCpBatchAvg, t_out=val_outs, t_balanced_gcd=valBalancedGcd)

    
    def _evaluate(self, outs, labels, mask, superpop, granularpop, \
        runAvgObj, cpRunAvgObj, gcdBalancedMetricsObj, loss_inner):
        
        BatchAvg, CpBatchAvg = self._evaluateAccuracy(outs, labels, mask, \
        batchLoss={'loss_main':loss_inner.loss_main, \
        'loss_aux':loss_inner.loss_aux}, batchCpLoss={'loss_cp':loss_inner.loss_cp}, \
        runAvgObj=runAvgObj, cpRunAvgObj=cpRunAvgObj)

        BalancedGcd=None
        if self.params.geography and self.params.evalBalancedGcd:
            gcdMatrix=GcdLoss().rawGcd(outs.coord_main, labels.coord_main).detach()
            BalancedGcd=self.getExtraGcdMetrics(gcdBalancedMetricsObj, gcdMatrix, superpop, granularpop, mask)
        return BatchAvg, CpBatchAvg, BalancedGcd

    # Need to think this one
    def _evaluateAccuracy(self, y, target, mask,**kwargs):
        runAvgObj=kwargs.get('runAvgObj')
        cpRunAvgObj=kwargs.get('cpRunAvgObj')
        batchLoss=kwargs.get('batchLoss')
        batchCpLoss=kwargs.get('batchCpLoss')
        # sample_size=kwargs.get('sample_size')
        cpThresh=kwargs.get('cpThresh')
        
        sample_size = torch.sum(mask)
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

    def getBalancedClassGcd(self, gcdObj):
        balancedGcdMetrics={}
        meanBalancedGcd=self.option['balancedMetrics']['meanBalanced'](gcdObj)()
        medianBalancedGcd=self.option['balancedMetrics']['medianBalanced'](gcdObj)()
        balancedGcdMetrics['meanBalancedGcdSp'], balancedGcdMetrics['meanBalancedGcdGp'] = meanBalancedGcd
        balancedGcdMetrics['medianBalancedGcdSp'], balancedGcdMetrics['medianBalancedGcdGp'] = medianBalancedGcd
        balancedGcdMetrics['median']=self.option['balancedMetrics']['median'](gcdObj)()
        return  balancedGcdMetrics

    def getExtraGcdMetrics(self, gcdMetricsObj, gcdMatrix, superpop, granularpop, mask):       
        gcdMetricsObj.fillData(gcdMatrix*mask.squeeze(-1))
        gcdMetricsObj.balancedMetric(superpop, granularpop)
        trainBalancedGcd = self.getBalancedClassGcd(gcdMetricsObj)
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

    def _plotSample(self, **kwargs):
        idx=kwargs.get('idx')
        idxSample = kwargs.get('idxSample')
        idxLabel = kwargs.get('idxLabel')
        vcf_idx= kwargs.get('idxVcf_idx')
        fig, fig_geo = self.plotObj.plot_index(idxSample, idxLabel, vcf_idx)
        phase = kwargs.get('phase')
        # phase="train" if any([m.training for m in list(self.model.values())]) else "valid/test"
        self.wandb.log({f" Image for idx {idx} for {phase}":self.wandb.Image(fig)})
        if self.params.geography:
            self.wandb.log({f" Image for idx {idx} for {phase} geography":fig_geo})

    def _logger(self, **kwargs):
        batch_num=kwargs.get('batch_num')
        batchAvg=kwargs.get('batchAvg')
        phase = kwargs.get('phase')
        # phase="train" if any([m.training for m in list(self.model.values())]) else "valid/test"
        self.wandb.log({f"MainTask_Loss/{phase}":batchAvg["loss_main"], "batch_num":batch_num})
        self.wandb.log({f"AuxTask_Loss/{phase}":batchAvg.get("loss_aux"), "batch_num":batch_num})

    def _checkModelParamGrad(self):
        print("**"*20+f" model {self.model.__class__.__name__}"+"**"*20)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None: print(n, p.grad) 
                else: print(f"{n}, None")

    def _logOutput(self, batchAvg, batch_num, vcf_idx, out, label):
        self._logger(batchAvg=batchAvg, batch_num=batch_num)
        if self.plotObj is not None : 
            # idx = np.random.choice(val_x.shape[0],1)[0]
            idx=30
            idxSample, idxLabel, idxVcf_idx = self._getSample(out=out, label=label \
                , vcf_idx=vcf_idx, idx=idx)
            self._plotSample(idxSample=idxSample, idxLabel=idxLabel \
                , idx=idx, idxVcf_idx=idxVcf_idx)

    @timer
    def launchTraining(self, modelSavePath, training_generator, validation_generator, **kwargs):
        test_generator = kwargs.get("test_generator")
        self.plotObj=kwargs.get("plotObj")
        self.wandb=kwargs.get("wandb")
        trial=kwargs.get('trial')
        
        modelOptimParams=self.model.getOptimizerParams()
        optimizer = torch.optim.Adam(modelOptimParams)
        
        # learning rate scheduler
        exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = self.params.lr_steps_decay,\
        verbose=True)
        # attention_models_ls=["Model_F", "Model_H"]
        # if self.params.model in attention_models_ls:
        #     optimizer = custom_opt(optimizer, d_model=self.params.att_input_size, \
        #     warmup_steps=self.params.att_warmup_steps, factor=self.params.att_factor, groups=self.params.warmup_lr_groups)
        print(("Begin Training...."))
        start_epoch = 0
        patience = 0
        for epoch in range(start_epoch, self.params.num_epochs):
            train_result = self.batchLoopTrain(optimizer, training_generator)
            eval_result = self.batchLoopValid(validation_generator)
            plt.close('all')
            
            # every step in the scheduler is per epoch
            exp_lr_scheduler.step(eval_result.t_accr['loss_main'])
            
            # logic for best model
            is_best = False
            if (epoch==start_epoch) or (eval_result.t_accr['loss_main'] < best_val_accr):
                best_val_accr = eval_result.t_accr['loss_main']
                is_best = True
            
            if epoch!=start_epoch:
                patience = early_stopping(eval_result.t_accr['loss_main'], val_prev_accr, patience, self.params.thresh)
                if patience == self.params.early_stopping_thresh:
                    print("Early stopping...")
                    break
            
            val_prev_accr = eval_result.t_accr['loss_main']

            if self.wandb is not None:
                self.epoch_logger("train", train_result, epoch)
                self.epoch_logger("valid", eval_result, epoch)
                self.wandb.log({f"val best accuracy":best_val_accr, "epoch":epoch})

            # saving a model at every epoch
            print(f"Saving at epoch {epoch}")
            print(f"train accr: {train_result.t_accr['loss_main']}, val accr: {eval_result.t_accr['loss_main']}")
            checkpoint = osp.join(modelSavePath, 'models_dir')
            models_state_dict = self.model.state_dict() 

            if not self.params.evaluateTest: test_result=t_results()
            if self.params.evaluateTest and epoch%20==0:
                test_result = self.batchLoopValid(test_generator)
                plt.close('all')
                if self.wandb is not None: self.epoch_logger("test", test_result, epoch)

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': models_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accr': eval_result._asdict(),
                'train_accr': train_result._asdict(),
                'test_accr': test_result._asdict()
                }, checkpoint, is_best=is_best)
            
            try:
                if epoch==start_epoch: 
                    self.params.save(''.join([modelSavePath, '/params.json']))
                    print(f"saving params at epoch:{epoch}")
            except Exception as e:
                print(f"exception while saving params:{e}")
                pass

            if self.params.hyper_search_type=='optuna':    
                trial.report(eval_result.t_accr['loss_main'], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                return best_val_accr
        #============================= Epoch Loop ===============================#    
        torch.cuda.empty_cache()

    def epoch_logger(self, phase, result, epoch):
        self.wandb.log({f"{phase}_metrics":result.t_accr, "epoch":epoch})
        if self.params.geography:
            self.wandb.log({f"{phase}_metrics":result.t_balanced_gcd, "epoch":epoch})
        if self.params.cp_predict:
            self.wandb.log({f"{phase}_metrics":result.t_cp_accr, "epoch":epoch})