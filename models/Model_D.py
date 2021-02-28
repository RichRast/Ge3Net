import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from collections import namedtuple

from helper_funcs import activate_mc_dropout, split_batch, square_normalize, get_gradient, Running_Average, form_mask
from evaluation import SmoothL1Loss, Weighted_Loss, gradient_reg, eval_cp_batch


cp_accr = namedtuple('cp_accr', ['cp_loss', 'Precision', 'Recall', 'Balanced_Accuracy'])
accr = namedtuple('accr', ['l1_loss', 'mse_loss', 'smoothl1_loss', 'weighted_loss', 'cp_accr'])
accr.__new__.defaults__=(None,)*len(accr._fields)
results = namedtuple('results', ['accr', 'pred'])

class model_D(object):
    def __init__(self, *args, params):
        self.aux_network = args[0]
        self.main_network = args[1]
        self.cp_network = args[2]
        self.params = params
        self.criterion = Weighted_Loss(alpha=self.params.weightLoss_alpha)
        self.smooth_l1_accr = SmoothL1Loss(reduction='sum', beta=self.params.SmoothLoss_beta)
        self.weighted_loss_accr = Weighted_Loss(reduction='sum', alpha=self.params.weightLoss_alpha)
        
    def train(self, optimizer, training_generator, writer=None, wandb=None):

        self.aux_network.train()
        self.main_network.train()
        self.cp_network.train()
        
        cp_pred = None
        reg_loss = 0.0
        cp_pred_loss = 0.0
        accr_avg=[]
        for a in accr._fields:
            if a!='cp_accr':
                accr_avg.append(Running_Average(num_members=1))
            else:
                accr_avg.append(Running_Average(len(cp_accr._fields)))                
        for i, train_gen in enumerate(training_generator):
            preds, target = [], []
            train_x, train_y, cp_mask = train_gen
            train_x = train_x[:, 0:self.params.chmlen].float().to(self.params.device)
            train_labels = train_y.to(self.params.device)
            cp_mask = cp_mask.to(self.params.device) # mask for transition windows
            
            # Forward pass
            # update the gradients to zero
            optimizer.zero_grad()
            
            out1, out2, _, out4 = self.aux_network(train_x)
            
            loss_aux = self.criterion(out4*cp_mask, train_labels*cp_mask)
            train_x = out1.reshape(train_x.shape[0], self.params.n_win, self.params.aux_net_hidden)
            # add residual connection by taking the gradient of aux network predictions
            aux_diff = get_gradient(out4)
            train_lstm = torch.cat((train_x, aux_diff), dim =2)
            
            #to do make this dynamic so that each model class is defined with dynamic input and outputs
            self.params.rnn_net_in= train_lstm.shape[-1]

            if self.params.tbptt:
                rnn_state = None
                bptt_batch_chunks = split_batch(train_lstm.clone(), self.params.tbptt_steps)
                batch_cp_mask_chunks = split_batch(cp_mask, self.params.tbptt_steps)
                batch_label = split_batch(train_labels, self.params.tbptt_steps)
                
                for x_chunk, batch_label_chunk, cp_mask_chunk in zip(bptt_batch_chunks, batch_label, batch_cp_mask_chunks):
                    x_chunk = V(x_chunk)
                    
                    train_vec_64, train_vector, rnn_state = self.main_network(x_chunk, rnn_state)

                    # for each bptt size we have the same batch_labels
                    loss_main = self.criterion(train_vector*cp_mask_chunk, batch_label_chunk*cp_mask_chunk)
                    preds.append(train_vector*cp_mask_chunk)
                    target.append(batch_label_chunk*cp_mask_chunk)
                    sample_size = batch_label_chunk.shape[0] * batch_label_chunk.shape[1]
                    if self.params.reg_loss:
                        #add regularization loss
                        reg_loss = gradient_reg(self.params.cp_detect, train_vector*cp_mask_chunk, p = self.params.reg_loss_pow)
                    if self.params.cp_predict:
                        assert self.params.cp_detect, "cp detection is not true while cp prediction is true"
                        cp_pred_logits = self.cp_network(train_vec_64)
                        cp_pred = torch.round(torch.sigmoid(cp_pred_logits)) 
                        # invert the mask for the cp target
                        cp_target = torch.where(cp_mask_chunk[:,:,0].unsqueeze(2)==0.0, torch.tensor([1.0]).to(self.params.device), torch.tensor([0.0]).to(self.params.device))
                        cp_pred_loss = F.binary_cross_entropy_with_logits(cp_pred_logits, cp_target).item()
                        preds.append(cp_pred)
                        target.append(cp_target)
                    
                    loss_main += reg_loss + cp_pred_loss

                    # do back propagation for tbptt steps in time
                    loss_main.backward()

                    # after doing back prob, detach rnn state to implement TBPTT
                    # now rnn_state was detached and chain of gradients was broken
                    rnn_state = self.main_network._detach_rnn_state(rnn_state)
                    
            else:
                train_vec_64, train_vector, _ = self.main_network(train_lstm)
                preds.append(train_vector*cp_mask)
                target.append(train_labels*cp_mask)
                sample_size = train_labels.shape[0]*train_labels.shape[1]

                if self.params.reg_loss: # for regularization loss of the gradients of final prediction
                    reg_loss = gradient_reg(self.params.cp_detect, train_vector*cp_mask, p = self.params.reg_loss_pow)
                if self.params.cp_predict: # for parallel branch to predict changepoints
                    assert self.params.cp_detect, "cp detection is not true while cp prediction is true"
                    cp_pred_logits = self.cp_network(train_vec_64)
                    cp_pred = torch.round(torch.sigmoid(cp_pred_logits))    
                    cp_target = torch.where(cp_mask[:,:,0].unsqueeze(2)==0.0, torch.tensor([1.0]).to(self.params.device), torch.tensor([0.0]).to(self.params.device))
                    cp_pred_loss = F.binary_cross_entropy_with_logits(cp_pred_logits, cp_target).item()
                    preds.append(cp_pred)
                    target.append(cp_target)
                
                loss_main = self.criterion(train_vector*cp_mask, train_labels*cp_mask)
                
            y = [preds, target]
            train_accr = self.evaluate_accuracy(accr_avg, sample_size, accr, *y)

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
                #wandb.log({"train_cp_pred_logits":cp_pred_logits.detach().cpu(),"batch_num":i})
                #wandb.log({"train_cp_loss":train_accr.cp_accr.cp_loss,"batch_num":i})
        # preds for tbptt will need to have a separate logic
        train_result=results(accr = train_accr, pred=[None, cp_pred, None])
        
        # delete tensors for memory optimization
        del train_x, train_labels, cp_mask, out1, out2, out4, train_lstm, \
            aux_diff, loss_aux, loss_main, train_vec_64, train_vector

        if self.params.cp_predict:
            del cp_pred_logits, cp_pred, cp_target, cp_pred_loss

        torch.cuda.empty_cache()
        return train_result
        
    def valid(self, validation_generator, writer=None, wandb=None):        
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

            for j, val_gen in enumerate(validation_generator):
                val_x, val_y, _ = val_gen
                val_x = val_x[:, 0:self.params.chmlen].float().to(self.params.device)
                val_labels = val_y.to(self.params.device)
                #cp_mask = cp_mask.to(self.params.device)
                sample_size = val_labels.shape[0]*val_labels.shape[1]
                output_list, vec_64_list, cp_pred_list = [], [], []
                preds, target =[], []
                
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
                
                preds.append(outputs_cat)
                target.append(val_labels)

                
                if self.params.cp_predict:
                    cp_pred_out_logits = self.cp_network(vec_64_cat)
                    cp_pred_out = torch.round(torch.sigmoid(cp_pred_out_logits))
                    # cp_target = torch.where(cp_mask[:,:,0].unsqueeze(2)==0.0, torch.tensor([1.0]).to(self.params.device), torch.tensor([0.0]).to(self.params.device))
                    # cp_pred_loss = F.binary_cross_entropy_with_logits(cp_pred_out_logits, cp_target)
                    # preds.append(cp_pred_out)
                    # target.append(cp_target)
                
                val_loss_regress_MLP = self.criterion(out4, val_labels)
                val_loss_main = self.criterion(outputs_cat, val_labels)

                if j>0:
                    y_pred = torch.cat((y_pred, outputs_cat), dim=0)
                    y_pred_var = torch.cat((y_pred_var, outputs_var), dim=0)
                    cp_pred = torch.cat((cp_pred, cp_pred_out), dim=0)

                else:
                    y_pred = outputs_cat
                    y_pred_var = outputs_var
                    cp_pred = cp_pred_out
                   
                y = [preds, target]
                accuracy = self.evaluate_accuracy(accr_avg, sample_size, accr, *y) 

                if writer:
                    # write to Tensorboard
                    writer.add_scalar('MainTask_Loss/val', val_loss_main, j)
                    writer.add_scalar('AuxTask_Loss/val', val_loss_regress_MLP, j)
                    
                if wandb:
                    wandb.log({"MainTask_Loss/val":val_loss_main,"batch_num":j})
                    wandb.log({"AuxTask_Loss/val":val_loss_regress_MLP,"batch_num":j})
                    # wandb.log({"val_cp_pred_logits":cp_pred_out_logits.detach().cpu(),"batch_num":j})
                    # wandb.log({"val_cp_loss":accuracy.cp_accr.cp_loss,"batch_num":j})
                    
            eval_result = results(accr = accuracy, pred = [y_pred, cp_pred, y_pred_var])

        # delete tensors for memory optimization
        del val_x, val_labels, out1, out2, out4, val_x_sample, aux_diff, val_lstm, y_pred, y_pred_var,\
            vec_64, val_outputs, outputs_cat, outputs_var, vec_64_cat, val_loss_regress_MLP, val_loss_main
        
        if self.params.cp_predict:
            del cp_pred_out_logits, cp_pred_out, cp_pred

        torch.cuda.empty_cache()

        return eval_result
    
    def pred(self, data_generator):
        pass

    def evaluate_accuracy(self, accr_avg, sample_size, accr, *y):
    
        preds = y[0]
        target = y[1]
        
        pred_y = preds[0]
        target_y = target[0]
        assert len(preds)==len(target), "Target and pred variable length must be equal"

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
        # if self.params.cp_predict:
        #     cp_pred = preds[1]
        #     cp_target = target[1]
        #     cp_loss_sum = F.binary_cross_entropy(cp_pred, cp_target, reduction = 'sum')
        #     precision, recall, _, _, balanced_accuracy = eval_cp_batch(cp_pred, cp_target, seq_len = cp_target.shape[0])
        #     accr_avg[4].update([cp_loss_sum.detach().cpu().numpy(),precision, recall, balanced_accuracy] , [sample_size]*len(cp_accr._fields))
        #     cp_accuracy = accr_avg[4]()
          
        accuracy = accr(l1_loss, mse_loss, smooth_l1_loss, weighted_loss, cp_accr(*cp_accuracy))
        
        return accuracy

        
