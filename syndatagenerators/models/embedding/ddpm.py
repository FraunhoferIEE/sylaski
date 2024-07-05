import torch
import torch.nn as nn

class BayesianMtlEmbedding(nn.Module):
    def __init__(self, n_tasks, embedding_dim=256, kl_weight=1e-3, device=torch.device('cuda')):
        super().__init__()
        self.n_tasks = n_tasks
        self.embedding_dim = embedding_dim
        self.kl_weight = kl_weight
        self.device = device
        self.mu_embed = nn.Embedding(n_tasks, embedding_dim)
        self.log_var_embed = nn.Embedding(n_tasks, embedding_dim)
        self.kl_loss = 0.0
        self.criterion_history = []
        self.kl_history = []
        self.init_embeddings()

    def init_embeddings(self):
        self.mu_embed.weight.data = torch.randn_like(self.mu_embed.weight.data)
        self.log_var_embed.weight.data = torch.zeros_like(self.log_var_embed.weight.data) - 1.
  
    def forward(self, x_cat, train_flag=True):
        mu = self.mu_embed(x_cat)
        if train_flag:
            log_var = self.log_var_embed(x_cat)
            self.kl_loss = torch.mean(torch.exp(log_var) + mu**2 - 1 - log_var)
            std = torch.exp(0.5*log_var)
            noise = torch.randn_like(std)
            e = mu + std*noise
        else:
            e = mu 
        e = e.flatten(2)
        return e
    
    def getKLLoss(self):
        return self.kl_weight*self.kl_loss
    