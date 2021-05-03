import torch
from torch.autograd import Variable as V
import sys
sys.path.insert(1, '/home/users/richras/Ge2Net_Repo')
from models import distributions
from models import BOCD
import numpy as np
import pandas as pd
import allel
from helper_funcs import get_recomb_rate, interpolate_genetic_pos, form_windows,\
activate_mc_dropout, split_batch, square_normalize
from models import BOCD, distributions

class model_J(object):
    def __init__(self, aux_network, main_network):

        self.aux_network = aux_network
        self.main_network = main_network
        self.criterion = torch.nn.MSELoss()
        self.accr_criterion = torch.nn.MSELoss(reduction='sum')

    def train(self, optimizer, training_generator, params, writer=None):
        self.aux_network.train()
        self.main_network.train()
        accr = []
        total_samples = []

        for i, train_gen in enumerate(training_generator):

            train_x, train_y = train_gen

            train_x = train_x[:, 0:params.dataset['chmlen']].float().to(params.device)
            train_labels = train_y.to(params.device)

            # Forward pass
            # update the gradients to zero
            optimizer.zero_grad()
            
            out1, out7, _, out9 = self.aux_network(train_x)
            loss_aux = self.criterion(out9, train_labels)

            train_x = out1.reshape(train_x.shape[0], params.dataset['n_win'], params.conv['input'])

            train_vector = self.main_network(train_x)
            loss_main = self.criterion(train_vector, train_labels)
                
            tmp_accr_sum = self.accr_criterion(train_vector, train_labels)
            accr.append(tmp_accr_sum)
            
            sample_size = train_labels.shape[0]*train_labels.shape[1]
            total_samples.append(sample_size)
            
            # clip gradient norm
            #torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), params.clip)


            loss = loss_main + loss_aux
            loss.backward()

            # update the weights
            optimizer.step()
            
            if writer is not None:
                # Write to tensorboard for train every batch
                writer.add_scalar('MainTask_Loss/train', loss_main.item(), i)
                writer.add_scalar('AuxTask_Loss/train', loss_aux.item(), i)

        train_accr = torch.sum(torch.stack(accr)) / sum(total_samples)

        # plot_grad_flow(model.named_parameters())
        # plot_grad_flow_v2(model.named_parameters())  # grad flow plot
        
        return train_accr
    
    def eval(self, validation_generator, params, writer=None):
        self.main_network.eval()
        self.aux_network.eval()
        
        
        
        with torch.no_grad():
            accr = []
            total_samples = []
            
            for j, val_gen in enumerate(validation_generator):
                val_x, val_y = val_gen
                val_x = val_x[:, 0:params.dataset['chmlen']].float().to(params.device)
                val_labels = val_y.to(params.device)
                
                if params.mc_dropout:
                    activate_mc_dropout(self.main_network, self.aux_network)
                    output_list = []

                    for _ in range(params.mc_samples):
                        out1, out7, _ , out9 = self.aux_network(val_x)

                        val_loss_regress_MLP = self.criterion(out9, val_labels)

                        val_x_sample = out1.reshape(val_x.shape[0], params.dataset['n_win'], params.conv['input'])

                        val_outputs = self.main_network(val_x_sample)

                        output_list.append(val_outputs)
                    
                    
                    outputs_cat = torch.cat(output_list, 0).contiguous().\
                    view(params.mc_samples, -1, params.dataset['n_win'], val_outputs.shape[-1]).mean(0)
                else:
                    out1, out7, _ , out9 = self.aux_network(val_x)

                    val_loss_regress_MLP = self.criterion(out1, val_labels)

                    val_x_sample = out1.reshape(val_x.shape[0], params.dataset['n_win'], params.conv['input'])

                    _, val_outputs, _ = self.main_network(val_x_sample)
                    outputs_cat = val_outputs
                    
                val_loss_main = self.criterion(outputs_cat, val_labels)
                
                tmp_accr_sum = self.accr_criterion(outputs_cat, val_labels)
                accr.append(tmp_accr_sum)
                sample_size = val_labels.shape[0]*val_labels.shape[1]
                total_samples.append(sample_size)
                
                if j>0:
                    y_pred = torch.cat((y_pred, outputs_cat), dim=0)
                else:
                    y_pred = outputs_cat
                    
                if writer is not None:
                    # write to Tensorboard
                    writer.add_scalar('MainTask_Loss/val', val_loss_main.item(), j)
                    writer.add_scalar('AuxTask_Loss/val', val_loss_regress_MLP.item(), j)

            val_accr = torch.sum(torch.stack(accr)) / sum(total_samples)

        return val_accr, y_pred