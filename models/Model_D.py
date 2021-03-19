import torch
import numpy as np
from torch.autograd import Variable as V
import torch.nn.functional as F
from collections import namedtuple
from decorators import timer

from helper_funcs import activate_mc_dropout, split_batch, square_normalize, get_gradient, Running_Average, form_mask
from evaluation import SmoothL1Loss, Weighted_Loss, gradient_reg, eval_cp_batch, class_accuracy


cp_accr = namedtuple('cp_accr', ['cp_loss', 'Precision', 'Recall', 'Balanced_Accuracy'])
accr = namedtuple('accr', ['l1_loss', 'mse_loss', 'smoothl1_loss', 'weighted_loss', 'cp_accr', 'sp_accr'])
accr.__new__.defaults__=(None,)*len(accr._fields)
results = namedtuple('results', ['accr', 'pred'])

class model_D(object):
    def __init__(self, *args, params):
        self.aux_network = args[0]
        self.main_network = args[1]
        self.cp_network = args[2]
        self.sp_network = args[3]
        self.residual_network = args[4]
        self.params = params
        self.criterion = Weighted_Loss(alpha=self.params.weightLoss_alpha)
        self.smooth_l1_accr = SmoothL1Loss(reduction='sum', beta=self.params.SmoothLoss_beta)
        self.weighted_loss_accr = Weighted_Loss(reduction='sum', alpha=self.params.weightLoss_alpha)
        
    def train(self, optimizer, training_generator, plot_obj, writer=None, wandb=None):

        self.aux_network.train()
        self.main_network.train()
        self.cp_network.train()
        self.sp_network.train()
        self.residual_network.train()
        
        cp_pred = None
        reg_loss = 0.0
        cp_pred_loss = 0.0
        cp_pred_loss_chunk = 0.0
        sp_pred_loss_chunk = 0.0
        residual_loss_chunk = 0.0
        accr_avg=[]
        for a in accr._fields:
            if a!='cp_accr':
                accr_avg.append(Running_Average(num_members=1))
            else:
                accr_avg.append(Running_Average(len(cp_accr._fields)))                
        for i, train_gen in enumerate(training_generator):
            preds, target, y_pred_list, cp_pred_list, cp_target_list, cp_pred_logits_list, \
                sp_pred_list, residual_list = [[] for _ in range(8)]
            train_x, train_y, vcf_idx, cps, superpop = train_gen
            train_x = train_x[:, 0:self.params.chmlen].float().to(self.params.device)
            train_labels = train_y.to(self.params.device)
            cps = cps.to(self.params.device)
            cp_mask = (cps==0).float() # mask for transition windows

            if self.params.superpop_mask:
                sp_mask = self.get_superpop_mask(superpop, self.params.pop_num, self.params.dataset_dim)
            else:
                sp_mask = torch.ones_like(train_labels)

            superpop = superpop.to(self.params.device)
            sp_mask = sp_mask.to(self.params.device)

            # Forward pass
            # update the gradients to zero
            optimizer.zero_grad()
            
            out1, out2, _, out4 = self.aux_network(train_x)
            
            loss_aux = self.criterion(out4*cp_mask[:,:,0:3], train_labels[:,:,0:3]*cp_mask[:,:,0:3])
            train_x = out1.reshape(train_x.shape[0], self.params.n_win, self.params.aux_net_hidden)
            # add residual connection by taking the gradient of aux network predictions
            aux_diff = get_gradient(out4)
            train_lstm = torch.cat((train_x, aux_diff), dim =2)
            
            #to do make this dynamic so that each model class is defined with dynamic input and outputs
            self.params.rnn_net_in= train_lstm.shape[-1]

            if self.params.tbptt:
                rnn_state = None
                bptt_batch_chunks = split_batch(train_lstm.clone(), self.params.tbptt_steps)
                batch_cps_chunks = split_batch(cps, self.params.tbptt_steps)
                batch_label = split_batch(train_labels, self.params.tbptt_steps)
                sp_mask_chunks = split_batch(sp_mask, self.params.tbptt_steps)
                sp_chunks = split_batch(superpop, self.params.tbptt_steps)
                
                for x_chunk, batch_label_chunk, cps_chunk, sp_mask_chunk, sp_chunk in zip(bptt_batch_chunks, batch_label, batch_cps_chunks, sp_mask_chunks, sp_chunks):
                    x_chunk = V(x_chunk)
                    
                    train_vec_64, train_vector, rnn_state = self.main_network(x_chunk, rnn_state)
                    cp_mask_chunk = (cps_chunk==0).float()

                    # for each bptt size we have the same batch_labels
                    loss_main_chunk = self.criterion(train_vector*cp_mask_chunk[:,:,0:3], batch_label_chunk[:,:,0:3]*cp_mask_chunk[:,:,0:3])

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
                        residual_out_chunk = self.residual_network(train_vec_64)
                        residual_loss_chunk = self.criterion(residual_out_chunk*cp_mask_chunk[:,:,3:5]*sp_mask_chunk[:,:,3:5], \
                            batch_label_chunk[:,:,3:5]*cp_mask_chunk[:,:,3:5]*sp_mask_chunk[:,:,3:5])

                        residual_list.append(residual_out_chunk)
                        
                    loss_main_chunk += reg_loss + cp_pred_loss_chunk + sp_pred_loss_chunk + residual_loss_chunk

                    # do back propagation for tbptt steps in time
                    loss_main_chunk.backward()
                    
                    # after doing back prob, detach rnn state to implement TBPTT
                    # now rnn_state was detached and chain of gradients was broken
                    rnn_state = self.main_network._detach_rnn_state(rnn_state)
                    y_pred_list.append(train_vector)
                
                # concatenating across windows because training was done in chunks of windows
                y_pred = torch.cat(y_pred_list,1)
                loss_main = self.criterion(y_pred*cp_mask[:,:,0:3], train_labels[:,:,0:3]*cp_mask[:,:,0:3])
                preds.append(y_pred*cp_mask[:,:,0:3])
                target.append(train_labels[:,:,0:3]*cp_mask[:,:,0:3])
                sample_size = cp_mask[:,:,0:3].sum() 

                cp_pred_out = torch.cat(cp_pred_list,1)
                cp_target_out = torch.cat(cp_target_list, 1)
                cp_pred_out_logits =  torch.cat(cp_pred_logits_list, 1)
                cp_pred_loss = F.binary_cross_entropy_with_logits(cp_pred_out_logits, cp_target_out).item()
                preds.append(cp_pred_out_logits)
                target.append(cp_target_out)

                sp_pred_out = torch.cat(sp_pred_list,1)
                sp_pred_out = sp_pred_out.detach().cpu().numpy()
                superpop = superpop.detach().cpu().numpy()
                preds.append(sp_pred_out)
                target.append(superpop)

                residual_out = torch.cat(residual_list,1)                
                residual_loss = self.criterion(residual_out*cp_mask[:,:,3:5]*sp_mask[:,:,3:5], train_labels[:,:,3:5]*cp_mask[:,:,3:5]*sp_mask[:,:,3:5])

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

            # test for params.requires_grad
            # print("param requires grad for aux_network")
            # for param in self.aux_network.parameters():
            #     print(f"param : {param}, {param.requires_grad}")

            # print("param requires grad for aux_network")
            # for param in self.main_network.parameters():
            #     print(f"param : {param}, {param.requires_grad}")

            # print("param requires grad for aux_network")
            # for param in self.cp_network.parameters():
            #     print(f"param : {param}, {param.requires_grad}")

            # print("param requires grad for aux_network")
            # for param in self.sp_network.parameters():
            #     print(f"param : {param}, {param.requires_grad}")    

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
            if writer:
                # Write to tensorboard for train every batch
                writer.add_scalar('MainTask_Loss/train', loss_main, i)
                writer.add_scalar('AuxTask_Loss/train', loss_aux, i)
            if wandb:
                wandb.log({"MainTask_Loss/train":loss_main, "batch_num":i})
                wandb.log({"AuxTask_Loss/train":loss_aux, "batch_num":i})
                wandb.log({"train_cp_loss":cp_pred_loss,"batch_num":i})
                wandb.log({"train_cp_accr":train_accr.cp_accr._asdict(), "batch_num":i})
                wandb.log({"train_sp_pred":sp_pred_out, "batch_num":i})
                wandb.log({"train_residual_loss":residual_loss,"batch_num":i})

                # randomly or non-randomly select an index and plot the output
                if i==0:
                    idx = 71
                    y_target_idx = train_labels[idx,:,:].detach().cpu().numpy().reshape(-1, self.params.dataset_dim)
                    y_pred_idx = y_pred[idx,:,:].detach().cpu().numpy().reshape(-1, self.params.rnn_net_out)
                    y_subclass_idx = residual_out[idx,:,:].detach().cpu().numpy()
                    train_vcf_idx = vcf_idx[idx,:].detach().cpu().numpy().reshape(-1, 1)
                    fig, ax = plot_obj.plot_index(y_pred_idx, y_subclass_idx, y_target_idx, train_vcf_idx)
                    wandb.log({ f" Train Image for idx {idx} ":wandb.Image(fig)})
        
        # preds for tbptt will need to have a separate logic
        train_result=results(accr = train_accr, pred=[y_pred, cp_pred, None])
        
        # delete tensors for memory optimization
        del train_x, train_labels, cp_mask, out1, out2, out4, train_lstm, \
            aux_diff, loss_aux, loss_main, train_vec_64, train_vector

        if self.params.cp_predict:
            del cp_pred_logits, cp_pred, cp_target, cp_pred_loss

        torch.cuda.empty_cache()
        return train_result
        
    def valid(self, validation_generator, plot_obj, writer=None, wandb=None):        
        self.main_network.eval()
        self.aux_network.eval()
        self.cp_network.eval()
        self.sp_network.eval()
        self.residual_network.eval()
        
        with torch.no_grad():
            cp_pred_out = None
            accr_avg=[]
            for a in accr._fields:
                if a!='cp_accr':
                    accr_avg.append(Running_Average(num_members=1))
                else:
                    accr_avg.append(Running_Average(len(cp_accr._fields)))  

            for j, val_gen in enumerate(validation_generator):
                val_x, val_y, vcf_idx, cps, superpop = val_gen
                val_x = val_x[:, 0:self.params.chmlen].float().to(self.params.device)
                val_labels = val_y.to(self.params.device)
                cps = cps.to(self.params.device)
                cp_mask = (cps==0).float()

                if self.params.superpop_mask:
                    sp_mask = self.get_superpop_mask(superpop, self.params.pop_num, self.params.dataset_dim)
                else:
                    sp_mask = torch.ones_like(val_labels)

                superpop = superpop.to(self.params.device)
                sp_mask = sp_mask.to(self.params.device)

                sample_size = val_labels.shape[0]*val_labels.shape[1]*val_labels.shape[2]
                preds, target, output_list, vec_64_list, cp_pred_list = [[] for _ in range(5)]
                
                models = [self.aux_network, self.main_network, self.cp_network]
                if self.params.mc_dropout:
                    activate_mc_dropout(*models)
                else:
                    assert self.params.mc_samples==1, "MC dropout disabled"

                for _ in range(self.params.mc_samples):
                    out1, out2, _ , out4 = self.aux_network(val_x)
                    
                    val_x_sample = out1.reshape(val_x.shape[0], self.params.n_win, self.params.aux_net_hidden)
                    aux_diff = get_gradient(out4)

                    val_lstm = torch.cat((val_x_sample, aux_diff), dim =2)
                    vec_64, val_outputs, _ = self.main_network(val_lstm)
                    output_list.append(val_outputs)
                    vec_64_list.append(vec_64)
                    
                outputs_cat = torch.cat(output_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.n_win, val_outputs.shape[-1]).mean(0)
                
                outputs_var = torch.cat(output_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.n_win, val_outputs.shape[-1]).var(0)
                
                vec_64_cat = torch.cat(vec_64_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.n_win, vec_64.shape[-1]).mean(0)
                
                preds.append(outputs_cat*cp_mask[:,:,0:3])
                target.append(val_labels[:,:,0:3]*cp_mask[:,:,0:3])

                if self.params.cp_predict:
                    cp_pred_out_logits, cp_pred_out, cp_target, cp_pred_loss = self.get_cp_predict(vec_64_cat, cps)
                    preds.append(cp_pred_out_logits)
                    target.append(cp_target)
                if self.params.superpop_predict:
                    sp_pred_logits, sp_pred_out, sp_pred_loss= self.get_sp_predict(vec_64_cat, superpop)
                    preds.append(sp_pred_out.detach().cpu().numpy())
                    target.append(superpop.detach().cpu().numpy())
                if self.params.superpop_mask:
                    residual_out = self.residual_network(vec_64_cat)
                  
                val_loss_regress_MLP = self.criterion(out4*cp_mask[:,:,0:3], val_labels[:,:,0:3]*cp_mask[:,:,0:3])
                val_loss_main = self.criterion(outputs_cat*cp_mask[:,:,0:3], val_labels[:,:,0:3]*cp_mask[:,:,0:3])
                val_loss_residual = self.criterion(residual_out*cp_mask[:,:,3:5]*sp_mask[:,:,3:5], val_labels[:,:,3:5]*cp_mask[:,:,3:5]*sp_mask[:,:,3:5])
                
                # concatenating for all the batches for the particular epoch
                if j>0:
                    y_pred = torch.cat((y_pred, outputs_cat), dim=0)
                    y_pred_var = torch.cat((y_pred_var, outputs_var), dim=0)
                    cp_pred = torch.cat((cp_pred, cp_pred_out), dim=0)
                    sp_pred = torch.cat((sp_pred, sp_pred_out), dim=0)
                    residual_pred = torch.cat((residual_pred, residual_out), dim=0)

                else:
                    y_pred = outputs_cat
                    y_pred_var = outputs_var
                    cp_pred = cp_pred_out
                    sp_pred = sp_pred_out
                    residual_pred = residual_out
                   
                y = [preds, target]
                accuracy = self.evaluate_accuracy(accr_avg, sample_size, *y) 

                if writer:
                    # write to Tensorboard
                    writer.add_scalar('MainTask_Loss/val', val_loss_main, j)
                    writer.add_scalar('AuxTask_Loss/val', val_loss_regress_MLP, j)
                    
                if wandb:
                    wandb.log({"MainTask_Loss/val":val_loss_main,"batch_num":j})
                    wandb.log({"AuxTask_Loss/val":val_loss_regress_MLP,"batch_num":j})
                    wandb.log({"val_cp_loss":cp_pred_loss,"batch_num":j})
                    wandb.log({"val_cp_accr":accuracy.cp_accr._asdict(), "batch_num":j})
                    wandb.log({"val_sp_loss":sp_pred_loss, "batch_num":j})
                    wandb.log({"val_sp_pred":sp_pred.detach().cpu().numpy(), "batch_num":j})
                    wandb.log({"val_residual_loss":val_loss_residual,"batch_num":j})
                    
                    # randomly or non-randomly select an index and plot the output
                    if j==0:
                        idx = 30
                        y_target_idx = val_labels[idx,:,:].detach().cpu().numpy().reshape(-1, self.params.dataset_dim)
                        y_pred_idx = outputs_cat[idx,:,:].detach().cpu().numpy().reshape(-1, self.params.rnn_net_out)
                        y_subclass_idx = residual_pred[idx,:,:].detach().cpu().numpy()
                        val_vcf_idx = vcf_idx[idx,:].detach().cpu().numpy().reshape(-1, 1)
                        fig, ax = plot_obj.plot_index(y_pred_idx, y_subclass_idx, y_target_idx, val_vcf_idx)
                        wandb.log({ f"Val Image for idx {idx}":wandb.Image(fig)})
                        # print(f'sp_target:{superpop[idx,:]}')
                        # print(f'sp_pred:{sp_pred_out[idx,:]}')
                    
            eval_result = results(accr = accuracy, pred = [y_pred, cp_pred, y_pred_var])

        # delete tensors for memory optimization
        del val_x, val_labels, out1, out2, out4, val_x_sample, aux_diff, val_lstm, y_pred, y_pred_var,\
            vec_64, val_outputs, outputs_cat, outputs_var, vec_64_cat, val_loss_regress_MLP, val_loss_main
        
        if self.params.cp_predict:
            del cp_pred_out_logits, cp_pred_out, cp_pred

        torch.cuda.empty_cache()

        return eval_result
    
    @timer
    def pred(self, data_generator, wandb=None):   
        self.main_network.eval()
        self.aux_network.eval()
        self.cp_network.eval()
        
        with torch.no_grad():
            cp_pred_out = None
            accr_avg=[]
            for a in accr._fields:
                if a!='cp_accr':
                    accr_avg.append(Running_Average(num_members=1))
                else:
                    accr_avg.append(Running_Average(len(cp_accr._fields)))  

            for j, test_gen in enumerate(test_generator):
                x = test_gen
                x = x[:, 0:self.params.chmlen].float().to(self.params.device)
                output_list, vec_64_list, cp_pred_list = [], [], []
                
                models = [self.aux_network, self.main_network, self.cp_network]
                if self.params.mc_dropout:
                    activate_mc_dropout(*models)
                else:
                    assert self.params.mc_samples==1, "MC dropout disabled"

                for _ in range(self.params.mc_samples):
                    out1, out2, _ , out4 = self.aux_network(val_x)
                    
                    x_sample = out1.reshape(x.shape[0], self.params.n_win, self.params.aux_net_hidden)
                    aux_diff = get_gradient(out4)

                    test_lstm = torch.cat((x_sample, aux_diff), dim =2)
                    vec_64, test_outputs, _ = self.main_network(val_lstm)
                    output_list.append(test_outputs)
                    vec_64_list.append(vec_64)
                    
                outputs_cat = torch.cat(output_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.n_win, test_outputs.shape[-1]).mean(0)
                
                outputs_var = torch.cat(output_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.n_win, test_outputs.shape[-1]).var(0)
                
                vec_64_cat = torch.cat(vec_64_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.n_win, vec_64.shape[-1]).mean(0)
 

                if self.params.cp_predict:
                    cp_pred_out_logits = self.cp_network(vec_64_cat)
                    cp_pred_out = torch.round(torch.sigmoid(cp_pred_out_logits))
 
                if j>0:
                    y_pred = torch.cat((y_pred, outputs_cat), dim=0)
                    y_pred_var = torch.cat((y_pred_var, outputs_var), dim=0)
                    cp_pred = torch.cat((cp_pred, cp_pred_out), dim=0)

                else:
                    y_pred = outputs_cat
                    y_pred_var = outputs_var
                    cp_pred = cp_pred_out
                    
            # randomly or non-randomly select an index and plot the output

        test_result = results(accr = None, pred = [y_pred, cp_pred, y_pred_var])

        # delete tensors for memory optimization
        del x, out1, out2, out4, x_sample, aux_diff, test_lstm, y_pred, y_pred_var,\
            vec_64, test_outputs, outputs_cat, outputs_var, vec_64_cat
        
        if self.params.cp_predict:
            del cp_pred_out_logits, cp_pred_out, cp_pred

        torch.cuda.empty_cache()

        return test_result

    def evaluate_accuracy(self, accr_avg, sample_size, *y):
    
        [preds, target] = y
        
        pred_y = preds[0]
        target_y = target[0]
        assert len(preds)==len(target), "Target and pred variable length must be equal"
        assert pred_y.shape==target_y.shape, "Shapes for pred_y and target_y do not match"

        l1_loss_sum = F.l1_loss(pred_y, target_y, reduction='sum').item()
        # update the running avg accuracy
        accr_avg[0].update(l1_loss_sum, sample_size)
        # return the running avg accuracy
        l1_loss = accr_avg[0]()
        
        mse_loss_sum = F.mse_loss(pred_y, target_y, reduction='sum').item()
        accr_avg[1].update(mse_loss_sum, sample_size)
        mse_loss = accr_avg[1]()

        smooth_l1_loss_sum = self.smooth_l1_accr(pred_y, target_y, self.params.device)
        accr_avg[2].update(smooth_l1_loss_sum, sample_size)
        smooth_l1_loss = accr_avg[2]()

        weighted_loss_sum = self.weighted_loss_accr(pred_y, target_y)
        accr_avg[3].update(weighted_loss_sum, sample_size)
        weighted_loss = accr_avg[3]()
        
        cp_accuracy = [None]*len(cp_accr._fields)
        if self.params.cp_predict:
            cp_pred_out_logits = preds[1]
            cp_target = target[1]
            cp_loss_sum = F.binary_cross_entropy_with_logits(cp_pred_out_logits, cp_target, reduction = 'sum').item()
            cp_target = cp_target.squeeze(2)
            cp_pred = (torch.sigmoid(cp_pred_out_logits)>0.5).int()
            cp_pred = cp_pred.squeeze(2)
            precision, recall, _, _, balanced_accuracy = eval_cp_batch(cp_target, cp_pred)
            accr_avg[4].update([cp_loss_sum, precision, recall, balanced_accuracy] , [sample_size/target_y.shape[-1], 1, 1, 1])
            cp_accuracy = accr_avg[4]()

        sp_accuracy = None    
        if self.params.superpop_predict:
            sp_pred_out = preds[2]
            sp_target = target[2]
            sp_accr = class_accuracy(sp_pred_out, sp_target)
            accr_avg[5].update(sp_accr, 1)
            sp_accuracy = accr_avg[5]()

        accuracy = accr(l1_loss, mse_loss, smooth_l1_loss, weighted_loss, cp_accr(*cp_accuracy), sp_accuracy)
        
        return accuracy

    def get_superpop_mask(self, superpop, pop_num, dim):

        superpop = superpop.detach().cpu().numpy()
        sp_mask = np.zeros((superpop.shape[0], superpop.shape[1], dim))

        sp_mask[:,:,0:self.params.n_comp_overall] = 1

        for i, pop_num_val in enumerate(pop_num):
            if isinstance(pop_num_val, list):
                idx_subclass = np.nonzero(np.isin(superpop, pop_num_val))
            else:    
                idx_subclass = np.nonzero(superpop==pop_num_val)
            
            sp_mask[idx_subclass[0], idx_subclass[1], self.params.n_comp_overall+self.params.n_comp_subclass*i:self.params.n_comp_overall+self.params.n_comp_subclass*(i+1)]=1
        
        return torch.tensor(sp_mask).float().to(self.params.device)
    
    def get_cp_predict(self, x, cps):
        cp_pred_out_logits = self.cp_network(x)
        cp_pred_out = (torch.sigmoid(cp_pred_out_logits)>0.5).int()
        cp_target = cps[:,:,0].unsqueeze(2)
        cp_pred_loss = F.binary_cross_entropy_with_logits(cp_pred_out_logits, cp_target)

        return cp_pred_out_logits, cp_pred_out, cp_target, cp_pred_loss

    def get_sp_predict(self, x, sp_target):
        sp_pred_out_logits = self.sp_network(x)
        sp_pred_out_logits = sp_pred_out_logits.permute(0,2,1).contiguous()
        sp_target = sp_target.long()
        sp_pred_out = torch.argmax(F.log_softmax(sp_pred_out_logits,dim=1), dim =1).float()
        sp_pred_loss = F.cross_entropy(sp_pred_out_logits, sp_target)

        return sp_pred_out_logits, sp_pred_out, sp_pred_loss
    

        




        
