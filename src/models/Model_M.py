import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from collections import namedtuple

from helper_funcs import activate_mc_dropout, split_batch, square_normalize, get_gradient, Running_Average, form_mask
from evaluation import SmoothL1Loss, Weighted_Loss, gradient_reg, eval_cp_batch

cp_accr = namedtuple('cp_accr', ['cp_loss', 'Precision', 'Recall'])
accr = namedtuple('accr', ['l1_loss', 'mse_loss', 'smoothl1_loss', 'weighted_loss', 'cp_accr'])
accr.__new__.defaults__=(None,)*len(accr._fields)
results = namedtuple('results', ['accr', 'pred', 'weights'])

class model_M(object):
    def __init__(self, *args, params):

        self.aux_network = args[0]
        self.PE_network = args[1]
        self.transf_network = args[2]
        
        self.params = params
        self.criterion = Weighted_Loss(alpha=self.params.weightLoss_alpha)
        self.smooth_l1_accr = SmoothL1Loss(reduction='sum', beta=self.params.SmoothLoss_beta)
        self.weighted_loss_accr = Weighted_Loss(reduction='sum', alpha=self.params.weightLoss_alpha)
        
    def train(self, optim, training_generator, writer=None, wandb=None):
        self.aux_network.train()
        self.PE_network.train()
        self.transf_network.train()
        
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
            train_x = train_x[:, 0:self.params.dataset['chmlen']].float().to(self.params.device)
            train_labels = train_y.to(self.params.device)
            cp_mask = cp_mask.to(self.params.device) # mask for transition windows
            batch_size = train_labels.shape[0]*train_labels.shape[1]

            # Forward pass
            # update the gradients to zero
            optim.optimizer.zero_grad()
            
            out1, out2, _, out4 = self.aux_network(train_x)
            
            loss_aux = self.criterion(out4*cp_mask, train_labels*cp_mask)
            train_x = out1.reshape(train_x.shape[0], self.params.dataset['n_win'], self.params.aux_net['hidden_unit'])
            # add residual connection by taking the gradient of aux network predictions
            aux_diff = get_gradient(out4)
            input_x1 = torch.cat((train_x, aux_diff), dim =2)

            output_1= self.transf_network(self.PE_network(input_x1))
            
            preds.append(output_1*cp_mask)
            target.append(train_labels*cp_mask)

            if self.params.reg_loss: # for regularization loss of the gradients of final prediction
                reg_loss = gradient_reg(self.params, out4*cp_mask, p = self.params.reg_loss_pow)
            
            loss_main = self.criterion(output_1*cp_mask, train_labels*cp_mask)
                
            y = [preds, target]
            train_accr = self.evaluate_accuracy(accr_avg, batch_size, accr, *y)

            # backward pass
            loss = loss_main + loss_aux + reg_loss + cp_pred_loss
            loss.backward()

            # update the weights
            optim.step()
            
            #logging
            if writer:
                # Write to tensorboard for train every batch
                writer.add_scalar('MainTask_Loss/train', loss_main.item(), i)
                writer.add_scalar('AuxTask_Loss/train', loss_aux.item(), i)
            if wandb:
                wandb.log({"MainTask_Loss/train":loss_main.item(),"batch_num":i})
                wandb.log({"AuxTask_Loss/train":loss_aux.item(),"batch_num":i})
        # preds for tbptt will need to have a separate logic
        train_result=results(accr = train_accr, pred=None, weights=None)
        
        return train_result
        
    def valid(self, validation_generator, writer=None, wandb=None):
        self.aux_network.eval()
        self.PE_network.eval()
        self.transf_network.eval()
        
        with torch.no_grad():
            cp_pred_out = None
            accr_avg=[]
            for a in accr._fields:
                if a!='cp_accr':
                    accr_avg.append(Running_Average(num_members=1))
                else:
                    accr_avg.append(Running_Average(len(cp_accr._fields)))  

            for j, val_gen in enumerate(validation_generator):
                val_x, val_y, cp_mask = val_gen
                val_x = val_x[:, 0:self.params.dataset['chmlen']].float().to(self.params.device)
                val_labels = val_y.to(self.params.device)
                cp_mask = cp_mask.to(self.params.device)
                batch_size = val_labels.shape[0]*val_labels.shape[1]
                output_list, vec_64_list, cp_pred_list = [], [], []
                preds, target =[], []
                
                models = [self.aux_network, self.PE_network, self.transf_network]
                if self.params.mc_dropout:
                    activate_mc_dropout(*models)
                else:
                    assert self.params.mc_samples==1, "MC dropout disabled"

                for _ in range(self.params.mc_samples):
                    out1, out2, _ , out4 = self.aux_network(val_x)
                    val_x_sample = out1.reshape(val_x.shape[0], self.params.dataset['n_win'], self.params.aux_net['hidden_unit'])
                    aux_diff = get_gradient(out4)

                    input_x1 = torch.cat((val_x_sample, aux_diff), dim =2)
                    output_1 = self.transf_network(self.PE_network(input_x1))
                    
                    output_list.append(output_1)
                    
                outputs_cat = torch.cat(output_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.dataset['n_win'], output_1.shape[-1]).mean(0)
                
                outputs_var = torch.cat(output_list, 0).contiguous().\
                view(self.params.mc_samples, -1, self.params.dataset['n_win'], output_1.shape[-1]).var(0)
                
                preds.append(outputs_cat*cp_mask)
                target.append(val_labels*cp_mask)
                #att_weights_cat = torch.cat(att_weights_list, 0).contiguous().\
                #view(self.params.mc_samples, -1, self.params.dataset['n_win'], att_weights.shape[-1]).mean(0)
                
                val_loss_regress_MLP = self.criterion(out4*cp_mask, val_labels*cp_mask)
                val_loss_main = self.criterion(outputs_cat*cp_mask, val_labels*cp_mask)

                
                if j>0:
                    y_pred = torch.cat((y_pred, outputs_cat), dim=0)
                    y_pred_var = torch.cat((y_pred_var, outputs_var), dim=0)
                    # Todo: use the pred function to get all att_weights
                    # currently runs out of memory 
                    #att_weights_pred = torch.cat((att_weights_pred, att_weights_cat), dim=0)
                else:
                    y_pred = outputs_cat
                    y_pred_var = outputs_var
                    #att_weights_pred = att_weights_cat
                   
                y = [preds, target]
                accuracy = self.evaluate_accuracy(accr_avg, batch_size, accr, *y) 

                if writer:
                    # write to Tensorboard
                    writer.add_scalar('MainTask_Loss/val', val_loss_main.item(), j)
                    writer.add_scalar('AuxTask_Loss/val', val_loss_regress_MLP.item(), j)
                    
                if wandb:
                    wandb.log({"MainTask_Loss/val":val_loss_main.item(),"batch_num":j})
                    wandb.log({"AuxTask_Loss/val":val_loss_regress_MLP.item(),"batch_num":j})
                    
            eval_result = results(accr = accuracy, pred = [y_pred, cp_pred_out, y_pred_var], weights=None)
            
        return eval_result
    
    def pred(self, data_generator):
        pass

    def evaluate_accuracy(self, accr_avg, batch_size, accr, *y):
    
        preds = y[0]
        target = y[1]
        
        pred_y = preds[0]
        target_y = target[0]
        assert len(preds)==len(target), "Target and pred variable length must be equal"

        l1_loss_sum = F.l1_loss(pred_y, target_y, reduction='sum')
        # update the running avg accuracy
        accr_avg[0].update(l1_loss_sum.detach().cpu().numpy(), batch_size)
        # return the running avg accuracy
        l1_loss = accr_avg[0]()
        
        mse_loss_sum = F.mse_loss(pred_y, target_y, reduction='sum')
        accr_avg[1].update(mse_loss_sum.detach().cpu().numpy(), batch_size)
        mse_loss = accr_avg[1]()

        smooth_l1_loss_sum = self.smooth_l1_accr(pred_y, target_y, self.params.device)
        accr_avg[2].update(smooth_l1_loss_sum.detach().cpu().numpy(), batch_size)
        smooth_l1_loss = accr_avg[2]()

        weighted_loss_sum = self.weighted_loss_accr(pred_y, target_y)
        accr_avg[3].update(weighted_loss_sum.detach().cpu().numpy(), batch_size)
        weighted_loss = accr_avg[3]()
        
        cp_accuracy = [None]*len(cp_accr._fields)
        if len(preds)==2:
            cp_pred = preds[1]
            cp_target = target[1]
            cp_loss_sum = F.binary_cross_entropy(cp_pred, cp_target, reduction = 'sum')
            precision, recall = eval_cp_batch(cp_pred, cp_target)
            accr_avg[4].update([cp_loss_sum.detach().cpu().numpy(),precision, recall] , [batch_size]*len(cp_accr._fields))
            cp_accuracy = accr_avg[4]()
          
        accuracy = accr(l1_loss, mse_loss, smooth_l1_loss, weighted_loss, cp_accr(*cp_accuracy))
        
        return accuracy

        

