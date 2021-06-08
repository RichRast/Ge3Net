import torch
from torch import nn
from src.utils.decorators import timer
from src.utils.modelUtil import activate_mc_dropout, countParams
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import modelOuts, branchLoss
from src.models.AuxiliaryTask import AuxNetwork
from src.models.BasicBlock import logits_Block
import pdb
import snoop

class model_A(nn.Module):
    def __init__(self, params):
        super(model_A, self).__init__()
        self.params=params
        self.aux = AuxNetwork(params)
        self.cp = logits_Block(params) if self.params.cp_predict else None
        self._setOptimizerParams()

        for m in [self.aux, self.cp]:
            count_params=[]
            params_count=countParams(m)
            print(f"Parameter count for model {self.aux.__class__.__name__}:{params_count}")
            count_params.append(params_count)
        print(f"Total parameters:{sum(count_params)}")

    def _setOptimizerParams(self):
        self.Optimizerparams=[]
        for i, m in enumerate([self.aux, self.cp]):
            params_dict={}
            params_dict['params']= m.parameters()
            params_dict['lr'] = self.params.learning_rate[i]
            params_dict['weight_decay'] = self.params.weight_decay[i]
            self.Optimizerparams.append(params_dict)
    
    def getOptimizerParams(self):
        return self.Optimizerparams
        
    def forward(self, x, **kwargs):
        mask = kwargs.get('mask')
        mc_dropout = kwargs.get('mc_dropout')
        if mask is None: mask = 1.0

        # Run Aux Network
        def _forwardNet(x):
            out1, _, _, out4 = self.aux(x)
            out1 = out1.reshape(x.shape[0], self.params.n_win, self.params.aux_net_hidden)
        
            # add residual connection by taking the gradient of aux network predictions
            aux_diff = get_gradient(out4)
            out_nxt = torch.cat((out1, aux_diff), dim =2)
            out_aux = square_normalize(out4) if self.params.geography else out4
            outs = modelOuts(coord_main = out_aux*mask, coord_aux= out_aux*mask)
            return outs, out_nxt

        if mc_dropout is None:
            outs, out_nxt = _forwardNet(x)
        else:
            outs, out_nxt, coord_main_list = mc_dropout._run(_forwardNet, x)

        # Run CP Network
        cp_logits = self.cp(out_nxt) if self.cp is not None else None
        outs.cp_logits = cp_logits
        
        if mc_dropout is not None:
            return outs, coord_main_list
        return outs

    def _batch_train(self, **kwargs):
        train_x = kwargs.get('train_x')
        train_labels = kwargs.get('train_labels')
        mask = kwargs.get('mask')
        train_outs= self(train_x, mask=mask)
        loss_inner, lossBack = self._getLoss(train_outs, train_labels, mask = mask, phase = "train")
        return train_outs, loss_inner, lossBack

    def _batch_validate(self, **kwargs):
        mc_dropout = kwargs.get('mc_dropout')
        val_x = kwargs.get('val_x')
        val_labels = kwargs.get('val_labels')
        mask = kwargs.get('mask')

        if mc_dropout is not None: 
            activate_mc_dropout(*list(self.aux, self.cp))
        val_outs, val_outs_list = self(val_x, mc_dropout=mc_dropout)            
        loss_inner, _ = self._getLoss(val_outs, val_labels, mask=mask)
        return val_outs, val_outs_list, loss_inner

    def _getLoss(self, outs, target, **kwargs):
        mask = kwargs.get('mask')
        phase = kwargs.get('phase')
        if mask is None: mask = 1.0
        loss_aux = super()._getLoss(outs.coord_aux*mask, target.coord_main*mask)
        # loss_aux=self.criterion(outs.coord_aux*mask, target.coord_main*mask)
        loss_main=loss_aux
        loss_cp=None
        if self.cp is not None: 
            loss_cp = self.cp.loss(self.option['cpMetrics']['loss_cp'], \
                logits=outs.cp_logits, target=target.cp_logits)
        
        lossBack = None
        if phase == "train":
            sample_size=mask.sum()
            lossBack = loss_aux/sample_size
            if loss_cp is not None:
                lossBack += loss_cp/(target.cp_logits.shape[0]*target.cp_logits.shape[1])

        return branchLoss(loss_main=loss_aux, loss_aux=loss_main, loss_cp = loss_cp), lossBack


