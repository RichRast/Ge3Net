import torch
from torch import nn
from torch.autograd import Variable as V
from src.utils.decorators import timer
from src.utils.modelUtil import split_batch, countParams, activate_mc_dropout
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import branchLoss, modelOuts, RnnResults
from src.models.AuxiliaryTask import BaseNetwork
from src.models.Attention import AttentionBlock, PositionalEncoding
from src.models.BasicBlock import logits_Block

class model_K(nn.Module):
    def __init__(self, params, criterion, cp_criterion):
        super(model_K, self).__init__()
        self.params=params
        self.aux = BaseNetwork(self.params)
        self.pe = PositionalEncoding(self.params, self.params.aux_net_hidden)
        self.attentionBlock = AttentionBlock(self.params, self.params.aux_net_hidden, self.params.FFNN_output)
        self.cp = logits_Block(self.params, self.params.aux_net_hidden) if self.params.cp_predict else None
        self.criterion=criterion
        self.cp_criterion = cp_criterion if self.params.cp_predict else None
        self._setOptimizerParams()

        count_params=[]
        for m in [self.aux, self.attentionBlock, self.cp]:
            if m is not None:
                params_count=countParams(m)
                print(f"Parameter count for model {m.__class__.__name__}:{params_count}")
                count_params.append(params_count)
        print(f"Total parameters:{sum(count_params)}")

    def _setOptimizerParams(self):
        self.Optimizerparams=[]
        for i, m in enumerate([self.aux, self.attentionBlock, self.cp]):
            if m is not None:
                params_dict={}
                params_dict['params']= m.parameters()
                params_dict['lr'] = self.params.learning_rate[i]
                params_dict['weight_decay'] = self.params.weight_decay[i]
                self.Optimizerparams.append(params_dict)
    
    def getOptimizerParams(self):
        return self.Optimizerparams

    def forward(self, x, mask, **kwargs):        
        mc_dropout = kwargs.get('mc_dropout')

        # Run Aux and LSTM Network
        def _forwardNet(x):
            out1 = self.aux(x)
            #Todo - do we really need to reshape below?
            out1 = out1.reshape(x.shape[0], self.params.n_win, self.params.aux_net_hidden)
        
            
            out_att = self.attentionBlock(self.pe(out1))
            out_nxt=out1
            out_main = square_normalize(out_att) if self.params.geography else out_att
            outs = modelOuts(coord_main = out_main*mask)
            return outs, out_nxt

        if mc_dropout is None:
            outs, out_nxt = _forwardNet(x)
            outs.coord_mainLs=[outs.coord_main]
        else:
            outs, out_nxt, coord_mainLs = mc_dropout(_forwardNet, x)
            outs.coord_mainLs=coord_mainLs

        # Run CP Network
        cp_logits = self.cp(out_nxt) if self.cp is not None else None
        outs.cp_logits = cp_logits        
        return outs
    
    def _batch_train_1_step(self, train_x, train_labels, mask):
        train_outs= self(train_x, mask)
        loss_inner, lossBack = self._getLoss(train_outs, train_labels, mask)
        return train_outs, loss_inner, lossBack

    
    def _batch_validate_1_step(self, val_x, **kwargs):
        val_labels=kwargs.get('val_labels')
        mask=kwargs.get('mask')
        if mask is None: mask = torch.ones((val_x.shape[0], self.params.n_win, 1), device=self.params.device, dtype=torch.uint8)
        mc_dropout = kwargs.get('mc_dropout')
        if mc_dropout is not None: activate_mc_dropout(*[self.aux, self.attentionBlock, self.cp])
        val_outs = self(val_x, mask, mc_dropout=mc_dropout) #call forward
        if val_labels is None:
            return val_outs           
        loss_inner = self._getLoss(val_outs, val_labels, mask)
        return val_outs, loss_inner

    def _getLoss(self, outs, target, mask):
        loss_main=self.criterion(outs.coord_main*mask, target.coord_main*mask) if self.params.criteria!="gcd" \
        else self.criterion(outs.coord_main, target.coord_main, mask=mask) 

        loss_cp=None
        if self.cp is not None: 
            loss_cp = self.cp_criterion(outs.cp_logits, target.cp_logits, reduction='sum', \
            pos_weight=torch.tensor([self.params.cp_pos_weight]).to(self.params.device)).item()
        
        rtnLoss = branchLoss(loss_main=loss_main.item(), loss_cp = loss_cp)

        if self.training:
            sample_size=mask.sum()
            lossBack=loss_main/sample_size
            if loss_cp is not None: lossBack += loss_cp/(target.cp_logits.shape[0]*target.cp_logits.shape[1])
            return rtnLoss, lossBack
        return rtnLoss