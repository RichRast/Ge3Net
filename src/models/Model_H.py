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

class model_H(object):
    def __init__(self, aux_network, main_network, config, params):

        self.aux_network = aux_network
        self.main_network = main_network
        self.criterion = torch.nn.MSELoss()
        self.accr_criterion = torch.nn.MSELoss(reduction='sum')

        # compute recomb_rate
        genetic_map_path = config['data.genetic_map']
        vcf_file_path = config['data.vcf_dir']
        df_gm_chm, df_vcf, df_vcf_pos = get_recomb_rate(genetic_map_path, vcf_file_path, chm='chr22')
        df_snp_pos = interpolate_genetic_pos(df_vcf_pos, df_gm_chm)
        recomb_w = form_windows(df_snp_pos, params.dataset)
        self.recomb_rate = np.diff(recomb_w)

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
            
            _, out7, _, out9 = self.aux_network(train_x)
            loss_aux = self.criterion(out9, train_labels)

            train_x = out7.reshape(train_x.shape[0], params.dataset['n_win'], params.aux_net['hidden_unit'])

            #pass train_x through BOCD and pass the predictive probs (log_prob_x_given_x1)
            #as an additional feature to lstm
            # CPD is on CPU for now, Todo make them both to either device
            data_tensor = out9
            batch_size_cpd = out9.shape[0]
            n_vec_dim = out9.shape[-1]
            mu_prior = torch.mean(data_tensor, dim =1).float().reshape(batch_size_cpd, 1,n_vec_dim)
            cov_prior = (torch.var(data_tensor, dim =1).float().unsqueeze(1) * \
                         torch.eye(n_vec_dim).to(params.device)).reshape(batch_size_cpd,1,n_vec_dim,n_vec_dim)
            cov_x = torch.eye(n_vec_dim).to(params.device).unsqueeze(0).repeat([batch_size_cpd,1,1]).reshape(batch_size_cpd,1,n_vec_dim,n_vec_dim)
            
            likelihood_model = distributions.Multivariate_Gaussian(mu_prior, cov_prior, cov_x)
            T = params.dataset['n_win']
            
            model_cpd_train = BOCD.BOCD(self.recomb_rate, T, likelihood_model, batch_size_cpd)

            _,_, predictive, e_mean = model_cpd_train.run_recursive(data_tensor, params.device)
            #shape of predictive will be batch_size x 318(n_win + 1) x dim 
            train_lstm = torch.cat((train_x, predictive[:,:-1,:]), dim =2)


            # LSTM
            rnn_state = None
            bptt_batch_chunks = split_batch(train_lstm.clone(), params.rnn_net['tbptt_steps'])
            batch_label = split_batch(train_labels, params.rnn_net['tbptt_steps'])

            if params.rnn_net['tbptt']:
                for text_chunk, batch_label_chunk in zip(bptt_batch_chunks, batch_label):
                    text_chunk = V(text_chunk)

                    _, train_vector, rnn_state = self.main_network(text_chunk, rnn_state)

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
                _,_,train_vector, _ = self.main_network(train_x)
                loss_main = self.criterion(train_vector, train_labels)
                
                tmp_accr_sum = self.accr_criterion(train_vector, train_labels)
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
                        _, out7, _ , out9 = self.aux_network(val_x)

                        val_loss_regress_MLP = self.criterion(out9, val_labels)

                        val_x_sample = out7.reshape(val_x.shape[0], params.dataset['n_win'], params.aux_net['hidden_unit'])

                        #pass train_x through BOCD and pass the predictive probs (log_prob_x_given_x1)
                        #as an additional feature to lstm
                        # CPD is on CPU for now, Todo make them both to either device
                        data_tensor = out9
                        batch_size_cpd = out9.shape[0]
                        n_vec_dim = out9.shape[-1]
                        mu_prior = torch.mean(data_tensor, dim =1).float().reshape(batch_size_cpd, 1,n_vec_dim)
                        cov_prior = (torch.var(data_tensor, dim =1).float().unsqueeze(1) * \
                                    torch.eye(n_vec_dim).to(params.device)).reshape(batch_size_cpd,1,n_vec_dim,n_vec_dim)
                        cov_x  = torch.eye(n_vec_dim).to(params.device).unsqueeze(0).repeat([batch_size_cpd,1,1]).reshape(batch_size_cpd,1,n_vec_dim,n_vec_dim)

                        likelihood_model = distributions.Multivariate_Gaussian(mu_prior, cov_prior, cov_x)
                        T = params.dataset['n_win']

                        model_cpd_val = BOCD.BOCD(self.recomb_rate, T, likelihood_model, batch_size_cpd)

                        _,_, predictive,e_mean = model_cpd_val.run_recursive(data_tensor, params.device)
                        #shape of predictive will be batch_size x 318(n_win + 1) x dim 
                        val_lstm = torch.cat((val_x_sample, predictive[:,:-1,:]), dim =2)

                        _, val_outputs, _ = self.main_network(val_lstm)

                        output_list.append(val_outputs)
                    
                    
                    outputs_cat = torch.cat(output_list, 0).contiguous().\
                    view(params.mc_samples, -1, params.dataset['n_win'], val_outputs.shape[-1]).mean(0)
                else:
                    _, out7, _ , out9 = self.aux_network(val_x)

                    val_loss_regress_MLP = self.criterion(out9, val_labels)

                    val_x_sample = out7.reshape(val_x.shape[0], params.dataset['n_win'], params.aux_net['hidden_unit'])

                    #pass train_x through BOCD and pass the predictive probs (log_prob_x_given_x1)
                    #as an additional feature to lstm
                    # CPD is on CPU for now, Todo make them both to either device
                    data_tensor = out9
                    batch_size_cpd = out9.shape[0]
                    n_vec_dim = out9.shape[-1]
                    mu_prior = torch.mean(data_tensor, dim =1).float().reshape(batch_size_cpd, 1,n_vec_dim)
                    cov_prior = (torch.var(data_tensor, dim =1).float().unsqueeze(1) * \
                                 torch.eye(n_vec_dim).to(params.device)).reshape(batch_size_cpd,1,n_vec_dim,n_vec_dim)
                    cov_x  = torch.eye(n_vec_dim).to(params.device).unsqueeze(0).repeat([batch_size_cpd,1,1]).reshape(batch_size_cpd,1,n_vec_dim,n_vec_dim)

                    likelihood_model = distributions.Multivariate_Gaussian(mu_prior, cov_prior, cov_x)
                    T = params.dataset['n_win']

                    model_cpd_val = BOCD.BOCD(self.recomb_rate, T, likelihood_model, batch_size_cpd)

                    _,_, predictive = model_cpd_val.run_recursive(data_tensor, params.device)
                    #shape of predictive will be batch_size x 318(n_win + 1) x dim 
                    val_lstm = torch.cat((val_x_sample, predictive[:,:-1,:]), dim =2)

                    _, val_outputs, _ = self.main_network(val_lstm)
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