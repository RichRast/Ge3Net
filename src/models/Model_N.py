import torch
from torch.autograd import Variable as V
import sys
sys.path.insert(1, '/home/users/richras/Ge2Net_Repo')

from models import LSTM, AuxiliaryTask
from helper_funcs import activate_mc_dropout, split_batch, square_normalize

class model_N(object):
    def __init__(self, aux_network, main_network, main_network2, main_network3):

        self.aux_network = aux_network
        self.main_network = main_network
        self.main_network2 = main_network2
        self.main_network3 = main_network3
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

            train_x = out1.reshape(train_x.shape[0], params.dataset['n_win'], params.aux_net['hidden_unit'])
            aux_diff = out9[:,:-1,:]-out9[:,1:,:]
            #insert 0 at the end
            aux_diff = torch.cat((aux_diff, torch.zeros_like(aux_diff[:,0,:]).to(params.device).unsqueeze(1)), dim=1)
            
            train_lstm = torch.cat((train_x, aux_diff), dim =2)
            
            rnn_state = None
            bptt_batch_chunks = split_batch(train_lstm.clone(), params.rnn_net['tbptt_steps'])
            batch_label = split_batch(train_labels, params.rnn_net['tbptt_steps'])

            if params.rnn_net['tbptt']:
                for text_chunk, batch_label_chunk in zip(bptt_batch_chunks, batch_label):
                    text_chunk = V(text_chunk)
                    att_train_x, att_scores, weights = self.main_network2(text_chunk)
                    
                    _, train_vector, rnn_state = self.main_network(att_train_x, rnn_state)

                    # for each bptt size we have the same batch_labels
                    loss_main = self.criterion(train_vector, batch_label_chunk)

                    # do back propagation for tbptt steps in time
                    loss_main.backward()

                    # after doing back prob, detach rnn state to implement TBPTT
                    # now rnn_state was detached and chain of gradients was broken
                    rnn_state = self.main_network._detach_rnn_state(rnn_state)

                    
                    tmp_accr_sum = self.accr_criterion(train_vector, batch_label_chunk)
                    accr.append(tmp_accr_sum)
            else:
                
                train_vector, train_out, _ = self.main_network(train_lstm)
                att_train_x, att_scores, weights = self.main_network2(train_vector)
                train_vector2, train_out2,  _ = self.main_network3(att_train_x)
                loss_main = self.criterion(train_out2, train_labels)
                
                tmp_accr_sum = self.accr_criterion(train_out2, train_labels)
                accr.append(tmp_accr_sum)
            
            sample_size = train_labels.shape[0]*train_labels.shape[1]
            total_samples.append(sample_size)
            
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), params.clip)

            # backward pass
            if params.rnn_net['tbptt']:
                loss_aux.backward()
            else:
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
        
        activate_mc_dropout(self.main_network, self.aux_network)
        
        with torch.no_grad():
            accr = []
            total_samples = []
            
            for j, val_gen in enumerate(validation_generator):
                val_x, val_y = val_gen
                val_x = val_x[:, 0:params.dataset['chmlen']].float().to(params.device)
                val_labels = val_y.to(params.device)
                
                output_list = []
                #att_scores_ls = []
                #weights_ls =[]
                
                for _ in range(params.mc_samples):
                    out1, out7, _ , out9 = self.aux_network(val_x)
                    val_loss_regress_MLP = self.criterion(out9, val_labels)

                    val_x_sample = out1.reshape(val_x.shape[0], params.dataset['n_win'], params.aux_net['hidden_unit'])
                    aux_diff = out9[:,:-1,:]-out9[:,1:,:]
                    #insert 0 at the end
                    aux_diff = torch.cat((aux_diff, torch.zeros_like(aux_diff[:,0,:]).to(params.device).unsqueeze(1)), dim=1)

                    val_lstm = torch.cat((val_x_sample, aux_diff), dim =2)
                    
                    val_vector, val_outputs, _ = self.main_network(val_lstm)
                    att_val_x, att_scores, weights = self.main_network2(val_vector)
                    val_vector2, val_outputs2, _ = self.main_network3(att_val_x)
                    
                    output_list.append(val_outputs2)
                    #att_scores_ls.append(att_scores)
                    #weights_ls.append(weights)
                    
                    
                outputs_cat = torch.cat(output_list, 0).contiguous().\
                view(params.mc_samples, -1, params.dataset['n_win'], val_labels.shape[-1]).mean(0)
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

        return val_accr, y_pred, att_scores, weights
    
    def pred(self, data_generator, params):
        self.main_network.eval()
        self.aux_network.eval()
        
        activate_mc_dropout(self.main_network, self.aux_network)
        
        with torch.no_grad():            
            for j, data_gen in enumerate(data_generator):
                data_x = data_gen
                data_x = data_x[:, 0:params.dataset['chmlen']].float().to(params.device)
                
                output_list = []
                
                for _ in range(params.mc_samples):
                    _, out7, _ , out9 = self.aux_network(data_x)

                    x_sample = out7.reshape(data_x.shape[0], params.dataset['n_win'], params.rnn_net['input_size'])

                    _, outputs, _ = self.main_network(x_sample)
                    
                    output_list.append(outputs)
                    
                outputs_cat = torch.cat(output_list, 0).contiguous().\
                view(params.mc_samples,-1, params.dataset['n_win'], outputs.shape[-1]).mean(0)
                if j>0:
                    y_pred = torch.cat((y_pred, outputs_cat), dim=0)
                else:
                    y_pred = outputs_cat

        return y_pred



        