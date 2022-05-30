import torch
from torch import nn
from torch.autograd import Variable as V
from src.utils.decorators import timer
from src.utils.modelUtil import split_batch, countParams, activate_mc_dropout
from src.utils.dataUtil import square_normalize, get_gradient
from src.main.evaluation import branchLoss, modelOuts, RnnResults
from src.models.AuxiliaryTask import BaseNetwork
from src.models.LSTM import BiRNN
from src.models.BasicBlock import logits_Block

class model_B(nn.Module):
    def __init__(self, params, criterion, cp_criterion):
        super(model_B, self).__init__()
        self.params=params
        self.aux = BaseNetwork(self.params)
        self.lstm = BiRNN(self.params, self.params.aux_net_hidden, self.params.rnn_net_out)
        self.cp = logits_Block(self.params, self.params.rnn_net_hidden * (1+1*self.params.rnn_net_bidirectional)) if self.params.cp_predict else None
        self.criterion=criterion
        self.cp_criterion = cp_criterion if self.params.cp_predict else None
        self._setOptimizerParams()

        count_params=[]
        for m in [self.aux, self.lstm, self.cp]:
            params_count=countParams(m)
            print(f"Parameter count for model {m.__class__.__name__}:{params_count}")
            count_params.append(params_count)
        print(f"Total parameters:{sum(count_params)}")

    def _setOptimizerParams(self):
        self.Optimizerparams=[]
        for i, m in enumerate([self.aux, self.lstm, self.cp]):
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
        
            # add residual connection by taking the gradient of aux network predictions
            vec_64, out_rnn, _ = self.lstm(out1)
            out_nxt = vec_64
            out_main = square_normalize(out_rnn) if self.params.geography else out_rnn
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
        if self.params.tbptt:
            train_outs, loss_inner, lossBack = self._trainTbtt(train_x, train_labels, mask)
        else:
            train_outs= self(train_x, mask)
            loss_inner, lossBack = self._getLoss(train_outs, train_labels, mask)
        return train_outs, loss_inner, lossBack

    
    def _batch_validate_1_step(self, val_x, **kwargs):
        val_labels=kwargs.get('val_labels')
        mask=kwargs.get('mask')
        if mask is None: mask = torch.ones((val_x.shape[0], self.params.n_win, 1), device=self.params.device, dtype=torch.uint8)
        mc_dropout = kwargs.get('mc_dropout')
        if mc_dropout is not None: activate_mc_dropout(*[self.aux, self.lstm, self.cp])
        val_outs = self(val_x, mask, mc_dropout=mc_dropout) #call forward
        if val_labels is None:
            return val_outs           
        loss_inner = self._getLoss(val_outs, val_labels, mask)
        return val_outs, loss_inner

    def _tbtt(self, x, target):
        rnn_state = None
        bptt_batch_chunks = split_batch(x.clone(), self.params.tbptt_steps)
        batch_cps_chunks = split_batch(target.cp_logits, self.params.tbptt_steps)
        batch_label = split_batch(target.coord_main, self.params.tbptt_steps)
        loss_main_list, out_rnn_list, vec64_list=[],[],[]
        loss_cp_list, cp_logits_list = [], []
        for x_chunk, batch_label_chunk, cps_chunk in zip(bptt_batch_chunks, batch_label, batch_cps_chunks):
            x_chunk = V(x_chunk, requires_grad=True)
            
            vec_64, out_rnn_chunk, rnn_state = self.lstm(x_chunk, rnn_state)
            if self.params.geography: out_rnn_chunk=square_normalize(out_rnn_chunk)
            cp_logits = self.cp(vec_64) if self.cp is not None else None
            vec64_list.append(vec_64)
            out_rnn_list.append(out_rnn_chunk)
            cp_logits_list.append(cp_logits)
            
            # Calculate Loss
            cp_mask_chunk = (cps_chunk==0).float()
            loss_main_chunk=self.criterion(out_rnn_chunk*cp_mask_chunk, batch_label_chunk*cp_mask_chunk) \
            if self.params.criteria!="gcd" else self.criterion(out_rnn_chunk, batch_label_chunk, mask=cp_mask_chunk) 
            loss_main_list.append(loss_main_chunk.item())
            sample_size=cp_mask_chunk.sum()
            loss_main_chunk /=sample_size
            loss_cp=None
            if self.cp is not None: 
                loss_cp = self.cp_criterion(cp_logits, cps_chunk, reduction='sum', \
                pos_weight=torch.tensor([self.params.cp_pos_weight]).to(self.params.device))
                loss_main_chunk +=loss_cp/(cps_chunk.shape[0]*cps_chunk.shape[1])
                loss_cp_list.append(loss_cp.item())
           
            loss_main_chunk.backward()
            # after doing back prob, detach rnn state to implement TBPTT
            # now rnn_state was detached and chain of gradients was broken
            rnn_state = self.lstm._detach_rnn_state(rnn_state)
            
        loss_main = sum(loss_main_list)
        loss_cp = sum(loss_cp_list)
        out_rnn = torch.cat(out_rnn_list, 1).detach()
        vec_64 = torch.cat(vec64_list, 1).detach()
        cp_logits = torch.cat(cp_logits_list, 1).detach()
        outs = modelOuts(coord_main = out_rnn, cp_logits=cp_logits)
        loss_inner = branchLoss(loss_main=loss_main, loss_cp=loss_cp)
        return vec_64, outs, loss_inner
   
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

    def _trainTbtt(self, x, target, mask):
        # Aux Model
        out1= self.aux(x)
        #Todo - do we really need to reshape below?
        out1 = out1.reshape(x.shape[0], self.params.n_win, self.params.aux_net_hidden)
        
        # Tbtt
        out_nxt, outs, loss_inner = self._tbtt(out1, target)        
        # backward loss needs to be calculated only for loss_aux because loss_main and 
        # loss_cp were already backwarded during tbtt
        outs.coord_main = outs.coord_main*mask
        lossBack=None
        return outs, loss_inner, lossBack

    
