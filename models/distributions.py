import torch

class Multivariate_Gaussian(object):
    def __init__(self, mu_prior, cov_prior, cov_x):
        self.mu = mu_prior
        self.cov = cov_prior
        self.batch_size = mu_prior.shape[0]
        self.dim = mu_prior.shape[-1]
        self.cov_x = cov_x
        self.cov_prior = cov_prior
        self.mu_prior = mu_prior
        
    def get_log_pdf(self, x):
        mu_cpu = self.mu.detach().cpu()
        cov_cpu = self.cov.detach().cpu()
        x_cpu = x.detach().cpu()
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu_cpu, cov_cpu)
        return dist.log_prob(x_cpu.unsqueeze(1))

    def update_params(self,t,x):

        data_size = x.size

        # for simplicity assume cov_x = cov_prior = identity matrix
        temp_cov = torch.inverse(torch.inverse(self.cov_prior) + t*(torch.inverse(self.cov_x))).reshape(self.batch_size, 1, self.dim, self.dim) # shape 1x3x3
        self.cov = torch.cat((self.cov, temp_cov), dim=1) # append to previous cov 
        
        #update mu
        temp_mu = torch.matmul((self.mu.unsqueeze(2)),(torch.inverse(self.cov[:,:-1,:,:])))

        temp_mu += torch.matmul(x.reshape(-1,1,1,self.dim), torch.inverse(self.cov_x))

        temp_mu = torch.matmul(temp_mu, self.cov[:,1:,:,:])
        temp_mu = temp_mu.squeeze(2)
        
        self.mu = torch.cat((self.mu_prior, temp_mu), dim=1)

    def sample():
        # Todo using sample or rsample
        dist = torch.distributions.multivariate_normal.MultivariateNormal(self.mu, self.cov)
        return dist.sample()
    
    def get_predictive_params(self, t, rt_given_x1_t):
        return torch.sum(torch.exp(rt_given_x1_t).unsqueeze(2) * self.mu[:,:t+1,:], dim=1)