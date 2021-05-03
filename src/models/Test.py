import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from utils.decorators import timer

from utils.modelUtil import activate_mc_dropout, split_batch, \
    square_normalize, get_gradient, Running_Average
from main.evaluation import SmoothL1Loss, Weighted_Loss, GcdLoss, \
    gradient_reg, eval_cp_batch, results, accr, cp_accr

class Ge3Net(object):
    def __init__(self, *args, params):
        self.model=[]
        for m in args:
            self.model.append(m)
        self.params = params
        if params.geography: self.criterion = Weighted_Loss(alpha=self.params.weightLoss_alpha) 
        else: self.criterion = GcdLoss() 
        self.smooth_l1_accr = SmoothL1Loss(reduction='sum', beta=self.params.SmoothLoss_beta)
        self.weighted_loss_accr = Weighted_Loss(reduction='sum', alpha=self.params.weightLoss_alpha)

    @timer
    def train(self, optimizer, training_generator, wandb=None):
        loss=[]
        for m in self.model:
            m.train()
            loss.append(0.)
        
        cp_pred = None
        
        accr_avg=[]
        for a in accr._fields:
            if a!='cp_accr':
                accr_avg.append(Running_Average(num_members=1))
            else:
                accr_avg.append(Running_Average(len(cp_accr._fields)))  

        n_comp = self._get_n_comp(self)

        for i, train_gen in enumerate(training_generator):
            train_x, train_y, vcf_idx, cps, superpop = train_gen
            train_x = train_x[:, 0:self.params.chmlen].float().to(self.params.device)
            train_labels = train_y.to(self.params.device)
            cps = cps.to(self.params.device)
            cp_mask = (cps==0).float() # mask for transition windows

            # sp_mask = torch.ones_like(train_labels).to(self.params.device)
            superpop = superpop.to(self.params.device)
            
            # Forward pass
            # update the gradients to zero
            optimizer.zero_grad()
            
            out_aux, train_lstm = self.model[0](self, train_x, cp_mask)
            if self.params.geography: square_normalize(out_aux) 
            loss_aux = self.criterion(out_aux*cp_mask, train_labels*cp_mask)
            
            if self.params.tbptt:
                self._rnnTbtt(train_lstm, train_labels, cps, superpop)
                
            else:
                y_pred = self.rnn(train_lstm)
                loss_main = self.criterion(y_pred*cp_mask, train_labels*cp_mask)
                
            y = [preds, target]
            train_accr = self.evaluate_accuracy(accr_avg, sample_size, *y)
   
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.params.clip)
            
            # backward pass
            if self.params.tbptt:
                loss_aux.backward()
            else:
                loss = loss_main + loss_aux + reg_loss + cp_pred_loss
                loss.backward()

            # update the weights
            optimizer.step()
            
            #logging
            if wandb:
                self._logger()
                self._plotSample(idx=71)
        
        # preds for tbptt will need to have a separate logic
        train_result=results(accr = train_accr, pred=[y_pred, cp_pred, None])
        
        # delete tensors for memory optimization
        del train_x, train_labels, cp_mask, out1, out2, out4, train_lstm, \
            aux_diff, loss_aux, loss_main, train_vec_64, train_vector

        if self.params.cp_predict:
            del cp_pred_logits, cp_pred, cp_target, cp_pred_loss, cp_pred_out, cp_target_out, cp_pred_out_logits
        
        if self.params.superpop_predict:
            del sp_pred_logits, sp_pred, sp_pred_loss_chunk, sp_pred_list, sp_pred_out, superpop

        torch.cuda.empty_cache()
        return train_result

    @timer
    def valid(self, validation_generator, wandb=None):
        pass

    @timer
    def pred(self, data_generator):
        pass

    def _get_n_comp(self):
        if self.params.residual:
            n_comp = self.params.n_comp_overall 
        else:
            n_comp = self.params.n_comp_overall + self.params.n_comp_subclass
        return n_comp

    def _auxNet(self, x):
        out1, _, _, out4 = self.aux_network(x)
        x = out1.reshape(x.shape[0], self.params.n_win, self.params.aux_net_hidden)
        # add residual connection by taking the gradient of aux network predictions
        aux_diff = get_gradient(out4)
        x_nxt = torch.cat((x, aux_diff), dim =2)
        return out4, x_nxt

    def _rnnTbtt(self, x, labels, cps, superpop):
        rnn_state = None
        bptt_batch_chunks = split_batch(train_lstm.clone(), self.params.tbptt_steps)
        batch_cps_chunks = split_batch(cps, self.params.tbptt_steps)
        batch_label = split_batch(train_labels, self.params.tbptt_steps)
        # sp_mask_chunks = split_batch(sp_mask, self.params.tbptt_steps)
        sp_chunks = split_batch(superpop, self.params.tbptt_steps)
        
        for x_chunk, batch_label_chunk, cps_chunk, sp_chunk in zip(bptt_batch_chunks, batch_label, batch_cps_chunks, sp_chunks):
            x_chunk = V(x_chunk, requires_grad=True)
            train_vec_64, train_vector, rnn_state = self.main_network(x_chunk, rnn_state)
            cp_mask_chunk = (cps_chunk==0).float()

            if self.params.reg_loss:
                #add regularization loss
                reg_loss = gradient_reg(self.params.cp_detect, train_vector*cp_mask_chunk, p = self.params.reg_loss_pow)
            if self.params.cp_predict:
                assert self.params.cp_detect, "cp detection is not true while cp prediction is true"
                cp_pred_logits, cp_pred, cp_target, cp_pred_loss_chunk = self.get_cp_predict(train_vec_64, cps_chunk)

            if self.params.superpop_predict:
                sp_pred_logits, sp_pred, sp_pred_loss_chunk = self.get_sp_predict(train_vec_64, sp_chunk)
            
            if self.params.superpop_mask:
                sp_mask_chunk = self.get_superpop_mask(sp_pred, self.params.pop_num, self.params.dataset_dim)  
            else:
                sp_mask_chunk = torch.ones_like(batch_label_chunk).to(self.params.device)
            
            if self.params.residual:
                residual_out_chunk = self.residual_network(train_vec_64)
                residual_loss_chunk = self.criterion(residual_out_chunk*cp_mask_chunk[...,-self.params.n_comp_subclass:]*sp_mask_chunk[...,-self.params.n_comp_subclass:], \
                    batch_label_chunk[...,-self.params.n_comp_subclass:]*cp_mask_chunk[...,-self.params.n_comp_subclass:]*sp_mask_chunk[...,-self.params.n_comp_subclass:])
                
                                                        
            # for each bptt size we have the same batch_labels
            loss_main_chunk = self.criterion(train_vector[..., :n_comp]*cp_mask_chunk[..., :n_comp]*sp_mask_chunk[..., :n_comp], \
                batch_label_chunk[..., :n_comp]*cp_mask_chunk[..., :n_comp]*sp_mask_chunk[..., :n_comp])

            loss_main_chunk += reg_loss + cp_pred_loss_chunk + sp_pred_loss_chunk + residual_loss_chunk

            # do back propagation for tbptt steps in time
            loss_main_chunk.backward()
            
            # after doing back prob, detach rnn state to implement TBPTT
            # now rnn_state was detached and chain of gradients was broken
            rnn_state = self.main_network._detach_rnn_state(rnn_state)

    def _rnn(self, x):
        vec_64, y_pred, _ = self.main_network(x)

        if self.params.reg_loss: # for regularization loss of the gradients of final prediction
            reg_loss = gradient_reg(self.params.cp_detect, train_vector*cp_mask, p = self.params.reg_loss_pow)
        if self.params.cp_predict: # for parallel branch to predict changepoints
            assert self.params.cp_detect, "cp detection is not true while cp prediction is true"
            cp_pred_logits, cp_pred, cp_target, cp_pred_loss = self.get_cp_predict(train_vec_64, cps)


    def changePointNet(self):


    def get_loss(self):

    def _evaluateAccuracy(self, accr_avg, sample_size, accr, *y):

    def _plotSample(idx):
        # randomly or non-randomly select an index and plot the output
        y_target_idx = train_labels[idx,:,:].detach().cpu().numpy().reshape(-1, self.params.dataset_dim)
        y_pred_idx = y_pred[idx,:,0:self.params.n_comp_overall].detach().cpu().numpy().reshape(-1, self.params.n_comp_overall)
        if self.params.residual:
            y_subclass_idx = residual_out[idx,:,:].detach().cpu().numpy().reshape(self.params.n_win, -1)
        else:
            y_subclass_idx = y_pred[idx,:,-self.params.n_comp_subclass:].detach().cpu().numpy().reshape(self.params.n_win, -1)
        
        if self.params.superpop_predict:
            y_sp_idx = sp_pred_out[idx,:].detach().cpu().numpy().reshape(1,-1)
        else:
            y_sp_idx=None
        train_vcf_idx = vcf_idx[idx,:].detach().cpu().numpy().reshape(-1, 1)
        fig, ax = plot_obj.plot_index(y_pred_idx, y_subclass_idx, y_sp_idx, y_target_idx, train_vcf_idx)
        wandb.log({f" Train Image for idx {idx} ":wandb.Image(fig)})

    def _logger(wandb, **kwargs):
        wandb.log({"MainTask_Loss/train":loss_main, "batch_num":i})
        wandb.log({"AuxTask_Loss/train":loss_aux, "batch_num":i})
        wandb.log({"train_cp_loss":cp_pred_loss,"batch_num":i})
        wandb.log({"train_cp_accr":train_accr.cp_accr._asdict(), "batch_num":i})
        if self.params.residual:
            wandb.log({"train_residual_loss":residual_loss_chunk, "batch_num":i})
        if self.params.plotting:
            