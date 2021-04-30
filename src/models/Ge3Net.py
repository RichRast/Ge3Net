import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from collections import namedtuple

from helper_funcs import activate_mc_dropout, split_batch, square_normalize, get_gradient, Running_Average
from evaluation import SmoothL1Loss, Weighted_Loss, gradient_reg, eval_cp_batch, results, accr, cp_accr

class Ge3Net(object):
    def __init__(self, *args, params):
        self.model=[]
        for m in args:
            self.model.append(m)
        self.params = params
        self.criterion = Weighted_Loss(alpha=self.params.weightLoss_alpha)
        self.smooth_l1_accr = SmoothL1Loss(reduction='sum', beta=self.params.SmoothLoss_beta)
        self.weighted_loss_accr = Weighted_Loss(reduction='sum', alpha=self.params.weightLoss_alpha)

    @timer
    def train(self, optimizer, training_generator, wandb=None):
        for m in self.model:
            m.train()
        
        cp_pred = None
        
        accr_avg=[]
        for a in accr._fields:
            if a!='cp_accr':
                accr_avg.append(Running_Average(num_members=1))
            else:
                accr_avg.append(Running_Average(len(cp_accr._fields)))  

        if self.params.residual:
            n_comp = self.params.n_comp_overall 
        else:
            n_comp = self.params.n_comp_overall + self.params.n_comp_subclass

        for i, train_gen in enumerate(training_generator):
            reg_loss = 0.0
            cp_pred_loss = 0.0
            cp_pred_loss_chunk = 0.0
            sp_pred_loss_chunk = 0.0
            residual_loss_chunk = 0.0
            preds, target, y_pred_list, cp_pred_list, cp_target_list, cp_pred_logits_list, \
                sp_pred_list, residual_list = [[] for _ in range(8)]
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
            
            out1, out2, _, out4 = self.aux_network(train_x)
            
            loss_aux = self.criterion(out4[:,:,0:n_comp]*cp_mask[:,:,0:n_comp], train_labels[:,:,0:n_comp]*cp_mask[:,:,0:n_comp])
            train_x = out1.reshape(train_x.shape[0], self.params.n_win, self.params.aux_net_hidden)
            # add residual connection by taking the gradient of aux network predictions
            aux_diff = get_gradient(out4)
            train_lstm = torch.cat((train_x, aux_diff), dim =2)
            
            if self.params.tbptt:
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
                        cp_pred_list.append(cp_pred)
                        cp_target_list.append(cp_target)
                        cp_pred_logits_list.append(cp_pred_logits)

                    if self.params.superpop_predict:
                        sp_pred_logits, sp_pred, sp_pred_loss_chunk = self.get_sp_predict(train_vec_64, sp_chunk)
                        sp_pred_list.append(sp_pred)
                    
                    if self.params.superpop_mask:
                        sp_mask_chunk = self.get_superpop_mask(sp_pred, self.params.pop_num, self.params.dataset_dim)  
                    else:
                        sp_mask_chunk = torch.ones_like(batch_label_chunk).to(self.params.device)
                    
                    if self.params.residual:
                        residual_out_chunk = self.residual_network(train_vec_64)
                        residual_loss_chunk = self.criterion(residual_out_chunk*cp_mask_chunk[:,:,-self.params.n_comp_subclass:]*sp_mask_chunk[:,:,-self.params.n_comp_subclass:], \
                            batch_label_chunk[:,:,-self.params.n_comp_subclass:]*cp_mask_chunk[:,:,-self.params.n_comp_subclass:]*sp_mask_chunk[:,:,-self.params.n_comp_subclass:])
                        residual_list.append(residual_out_chunk)
                                                                
                    # for each bptt size we have the same batch_labels
                    loss_main_chunk = self.criterion(train_vector[:,:,0:n_comp]*cp_mask_chunk[:,:,0:n_comp]*sp_mask_chunk[:,:,0:n_comp], \
                        batch_label_chunk[:,:,0:n_comp]*cp_mask_chunk[:,:,0:n_comp]*sp_mask_chunk[:,:,0:n_comp])

                    loss_main_chunk += reg_loss + cp_pred_loss_chunk + sp_pred_loss_chunk + residual_loss_chunk

                    # do back propagation for tbptt steps in time
                    loss_main_chunk.backward()
                    
                    # after doing back prob, detach rnn state to implement TBPTT
                    # now rnn_state was detached and chain of gradients was broken
                    if torch.cuda.device_count() > 1:
                        rnn_state = self.main_network.module._detach_rnn_state(rnn_state)
                    else:
                        rnn_state = self.main_network._detach_rnn_state(rnn_state)
                    y_pred_list.append(train_vector)
                
                # concatenating across windows because training was done in chunks of windows
                y_pred = torch.cat(y_pred_list,1).detach()
                if self.params.residual:
                    residual_out = torch.cat(residual_list, 1)
                cp_pred_out = torch.cat(cp_pred_list,1).detach()
                cp_target_out = torch.cat(cp_target_list, 1).detach()
                cp_pred_out_logits =  torch.cat(cp_pred_logits_list, 1).detach()
                cp_pred_loss = F.binary_cross_entropy_with_logits(cp_pred_out_logits, cp_target_out).detach().item()


                if self.params.superpop_predict:
                    sp_pred_out = torch.cat(sp_pred_list,1).detach()
                    superpop = superpop.detach().cpu().numpy()

                if self.params.superpop_mask:
                    sp_mask = self.get_superpop_mask(sp_pred_out, self.params.pop_num, self.params.dataset_dim)  
                else:
                    sp_mask = torch.ones_like(train_labels).to(self.params.device)

                loss_main = self.criterion(y_pred[:,:,0:n_comp]*cp_mask[:,:,0:n_comp]*sp_mask[:,:,0:n_comp], train_labels[:,:,0:n_comp]*cp_mask[:,:,0:n_comp]*sp_mask[:,:,0:n_comp]).item()
                preds.append(y_pred[:,:,0:n_comp]*cp_mask[:,:,0:n_comp]*sp_mask[:,:,0:n_comp])
                target.append(train_labels[:,:,0:n_comp]*cp_mask[:,:,0:n_comp]*sp_mask[:,:,0:n_comp])
                preds.append(cp_pred_out_logits)
                target.append(cp_target_out)
                if self.params.superpop_predict:
                    preds.append(sp_pred_out.detach().cpu().numpy())
                    target.append(superpop)
                sample_size = (cp_mask[:,:,0:n_comp]*sp_mask[:,:,0:n_comp]).sum() 

            else:
                train_vec_64, y_pred, _ = self.main_network(train_lstm)
                preds.append(y_pred*cp_mask)
                target.append(train_labels*cp_mask)
                sample_size = cp_mask.sum()

                if self.params.reg_loss: # for regularization loss of the gradients of final prediction
                    reg_loss = gradient_reg(self.params.cp_detect, train_vector*cp_mask, p = self.params.reg_loss_pow)
                if self.params.cp_predict: # for parallel branch to predict changepoints
                    assert self.params.cp_detect, "cp detection is not true while cp prediction is true"
                    cp_pred_logits, cp_pred, cp_target, cp_pred_loss = self.get_cp_predict(train_vec_64, cps)
                    preds.append(cp_pred)
                    target.append(cp_target)
                
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
                wandb.log({"MainTask_Loss/train":loss_main, "batch_num":i})
                wandb.log({"AuxTask_Loss/train":loss_aux, "batch_num":i})
                wandb.log({"train_cp_loss":cp_pred_loss,"batch_num":i})
                wandb.log({"train_cp_accr":train_accr.cp_accr._asdict(), "batch_num":i})
                if self.params.residual:
                    wandb.log({"train_residual_loss":residual_loss_chunk, "batch_num":i})
                if self.params.plotting:
                    # randomly or non-randomly select an index and plot the output
                    if i==0:
                        idx = 71
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

    @timer
    def pred(self, data_generator):

    def _evaluateAccuracy(self, accr_avg, sample_size, accr, *y):

    def _plotSample():

    def _log():