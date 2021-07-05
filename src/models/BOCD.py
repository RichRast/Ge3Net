import torch
from src.utils.decorators import timer

class BOCD(object):
    def __init__(self, recomb_rate, T, likelihood_model, batch_size):
        self.T = T
        self.batch_size = batch_size
        self.cp = torch.zeros((batch_size, self.T+1))
        self.prob_cp = torch.zeros((batch_size, self.T+1))
        self.recomb_rate = recomb_rate
        self.likelihood_model = likelihood_model
        
    def get_hazard(self, t, run_length=0):
        
        # hazard function at first and last time step
        if (t==0|t==self.T-1):
            hazard = 0
        else:
            # hazard = self.recomb_rate[t]/100
            # hazard=6/self.T
            hazard = 2/self.T   
        return hazard
    
    @timer
    def run_recursive(self, data, device):
        
        # define p_r_given_x as rl X T with axis 0 for run lengths and axis 1 for time steps
        log_prob_r_given_x = torch.tensor([-float('inf')]).reshape(1,1,1).repeat([self.batch_size, self.T+1, self.T+1]).to(device)
        log_prob_r_joint_x = torch.tensor([-float('inf')]).reshape(1,1,1).repeat([self.batch_size, self.T+1, self.T+1]).to(device)
        log_prob_r_joint_x[:,0,0] = 0
        log_prob_r_given_x[:,0,0] = 0
        dim=data.shape[-1]
        log_p_xt1_given_xt = torch.tensor([-float('inf')]).reshape(1,1,1).repeat([self.batch_size, self.T+1, 1]).to(device)
        e_mean = torch.zeros((self.batch_size, self.T+1,dim))
        self.cp[:, 0] = 0
        
        for t in range(1, self.T+1):
            x = data[:,t-1,:]
            
            # get UPM probabilities for each possible value of run length 
            pred_probs = self.likelihood_model.get_log_pdf(x) 
            # pred_probs = pred_probs.clone().detach().requires_grad_(True).to(device)
            
            #predictive
            log_p_xt1_given_xt[:,t-1,:] = torch.logsumexp(log_prob_r_given_x[:,0:t,t-1] + pred_probs, dim =1).unsqueeze(1)
            
            # get hazard function for all run_lengths
            p_hazard = torch.tensor([(self.get_hazard(t-1, i)) for i in range(t)]).float().unsqueeze(0).repeat([self.batch_size,1]).to(device)
             
            # compute growth probability
            log_prob_r_joint_x[:,1:t+1,t] = log_prob_r_given_x[:,0:t,t-1] + pred_probs + torch.log(1-p_hazard)
             
            # compute changepoint probability
            log_prob_r_joint_x[:,0,t] = torch.logsumexp(log_prob_r_given_x[:,0:t,t-1] + pred_probs +\
                                                      torch.log(p_hazard), dim =1)
            
            # normalizer, sum along that time step column
            den = torch.logsumexp(log_prob_r_joint_x[:,0:t+1,t], dim =1)
            den = den.unsqueeze(1)

            # posterior
            log_prob_r_given_x[:,0:t+1,t] = log_prob_r_joint_x[:,0:t+1,t] - den
            posterior = log_prob_r_given_x

            # Update params
            self.likelihood_model.update_params(t, x)
            
            # compute expected mean and cov at each time step
            e_mean[:,t,:] = self.likelihood_model.get_predictive_params(t-1, log_prob_r_given_x[:,:t, t-1])
            

            self.cp[:,t] = torch.argmax(log_prob_r_given_x[:,0:t+1,t], dim =1)
            self.prob_cp[:,t] = torch.exp(log_prob_r_given_x[:,4,t])

                 
        return posterior, log_prob_r_joint_x, log_p_xt1_given_xt, e_mean