import torch
from torch.autograd import Variable as V
import torch.nn as nn
import random
import sys
sys.path.insert(1, '/home/users/richras/Ge2Net_Repo')

from helper_funcs import activate_mc_dropout, split_batch, square_normalize

class model_G(object):
    def __init__(self, aux_network, main_network, params):

        self.aux_network = aux_network
        self.main_network = main_network
        self.criterion = torch.nn.MSELoss()
        self.accr_criterion = torch.nn.MSELoss(reduction='sum')
        self.input_size = params.seq2seq['input_size']
        self.hidden_size = params.seq2seq['hidden_size']
        self.output = params.seq2seq['output']
        
        self.n_win = params.dataset['n_win']
        self.num_layers = params.seq2seq['num_layers']
        self.teacher_forcing_ratio = params.seq2seq['teacher_forcing_ratio']
        self.dropout = params.seq2seq['dropout']
        self.device = params.device
        self.rnn = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True
            , bidirectional=True).to(self.device)
        self.fc1 = nn.Linear(self.hidden_size*2, self.output).to(self.device)
        
    def decoder(self, input_enc):     
        input_enc = input_enc.unsqueeze(1)
        out, (hidden, cell ) = self.rnn(input_enc)
        return self.fc1(out.squeeze(1)), out.squeeze(1), hidden, cell

    def train(self, optimizer, training_generator, params, writer=None):
        self.aux_network.train()
        self.main_network.train()
        accr = []
        total_samples = []

        for i, train_gen in enumerate(training_generator):

            train_x, train_y = train_gen

            train_x = train_x[:, 0:params.dataset['chmlen']].float().to(params.device)
            train_labels = train_y.to(params.device)
            
            # update the gradients to zero
            optimizer.zero_grad()
            
            # Forward pass
            _, out7, _, out9 = self.aux_network(train_x)
            loss_aux = self.criterion(out9, train_labels)

            train_x = out7.reshape(train_x.shape[0], params.dataset['n_win'], params.rnn_net['input_size'])
            
            rnn_state = None
            bptt_batch_chunks = split_batch(train_x.clone(), params.rnn_net['tbptt_steps'])
            batch_label = split_batch(train_labels, params.rnn_net['tbptt_steps'])

            # update the gradients to zero
            optimizer.zero_grad()
                        
            if params.rnn_net['tbptt']:
                j = 0
                for text_chunk, batch_label_chunk in zip(bptt_batch_chunks, batch_label):
                    text_chunk = V(text_chunk)
                    steps = text_chunk.shape[1]
                    
                    outputs = torch.zeros(text_chunk.shape[0], text_chunk.shape[1], self.output).to(self.device)
                    enc_out, train_vector, rnn_state = self.main_network(text_chunk, rnn_state)
                    
                    input_dec = enc_out[:,0,:]
                    
                    for t in range(1,steps ):

                        # take the last hidden and cell state to produce output and 
                        # next hidden and cell state
                        fc_output, dec_output, hidden, cell = self.decoder(input_dec)
                        
                        outputs[:,t,:] = fc_output

                        # decide whether to use teacher forcing or not
                        teacher_force = random.random() < self.teacher_forcing_ratio

                        # if teacher forcing, use the actual next label as next input
                        # if not, use the predicted input

                        input_dec = enc_out[:,t,:] if teacher_force else dec_output

                    
                    # for each bptt size we have the same batch_labels
                    loss_main = self.criterion(outputs, batch_label_chunk)
                    
                    # do back propagation for tbptt steps in time
                    loss_main.backward()

                    # after doing back prob, detach rnn state to implement TBPTT
                    # now rnn_state was detached and chain of gradients was broken
                    rnn_state = self.main_network._detach_rnn_state(rnn_state)

                    
                    tmp_accr_sum = self.accr_criterion(outputs, batch_label_chunk)
                    accr.append(tmp_accr_sum)
                    j +=1
            
            sample_size = train_labels.shape[0]*train_labels.shape[1]
            total_samples.append(sample_size)
            
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), params.clip)

            # backward pass
 
            loss = loss_aux 
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
                
                for _ in range(params.mc_samples):
                    _, out7, _ , out9 = self.aux_network(val_x)

                    val_loss_regress_MLP = self.criterion(out9, val_labels)

                    val_x_sample = out7.reshape(val_x.shape[0], params.dataset['n_win'], params.rnn_net['input_size'])

                    outputs = torch.zeros(val_x.shape[0], self.n_win, self.output).to(self.device)
                    enc_out, _, rnn_state = self.main_network(val_x_sample)
                    input_dec = enc_out[:,0,:]

                    for t in range(1, self.n_win):

                        # take the last hidden and cell state to produce output and 
                        # next hidden and cell state
                        fc_output, dec_output, hidden, cell = self.decoder(input_dec)
                        
                        outputs[:,t,:] = fc_output

                        # decide whether to use teacher forcing or not
                        teacher_force = random.random() < self.teacher_forcing_ratio

                        # if teacher forcing, use the actual next label as next input
                        # if not, use the predicted input

                        input_dec = enc_out[:,t,:] if teacher_force else dec_output

                    
                    
                    output_list.append(outputs)
                    
                    
                outputs_cat = torch.cat(output_list, 0).contiguous().\
                view(params.mc_samples, -1, params.dataset['n_win'], self.output).mean(0)
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



        