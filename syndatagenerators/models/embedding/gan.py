import torch
import torch.nn as nn

from tqdm import tqdm

class BayesianMtlEmbeddingGAN(nn.Module):
    def __init__(self, feature_dim ,n_tasks, embedding_dim=2, num_hidden=256, noise_dim=100,
                 device=torch.device('cuda')):
        super().__init__()
        self.mu_embed = nn.Embedding(n_tasks, embedding_dim)
        self.log_var_embed = nn.Embedding(n_tasks, embedding_dim)
        self.noise_dim = noise_dim
        self.device = device
        self.gen_history = []
        self.dis_history = []

        self.gen_net = nn.Sequential(
            nn.Linear(embedding_dim+feature_dim+noise_dim, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

        self.dis_net = nn.Sequential(
            nn.Linear(embedding_dim+feature_dim+1, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

    def noise(self, batch_size, device):
        return torch.randn((batch_size, 48, self.noise_dim)).to(device)
    
    def embedding(self, x_cat):
        with torch.no_grad():
            emb = self.mu_embed(x_cat.long()).flatten(2)
        return emb

    def generator(self, x_cat, x_cont):
        x_emb = self.embedding(x_cat)
        z = self.noise(x_cat.shape[0], x_cat.device)
        y = self.gen_net(torch.cat((x_emb, x_cont, z), dim=-1))
        return y
    
    def discriminator(self, x_cat, x_cont, y):
        x_emb = self.embedding(x_cat)
        d = self.dis_net(torch.cat((x_emb, x_cont, y), dim=-1))
        return d
    
    def train(self, dl_train, optimizer, criterion, n_epochs):
        gen_opt = optimizer['gen_opt']
        dis_opt = optimizer['dis_opt']
        self = self.to(self.device)
        for epoch in tqdm(range(n_epochs)):
            for x_cat, x_cont, y in dl_train:
                x_cat = x_cat.to(self.device)
                x_cont = x_cont.to(self.device)
                y = y.to(self.device)

                #train generator
                y_fake = self.generator(x_cat, x_cont)
                d_fake = self.discriminator(x_cat, x_cont, y_fake)
                loss_gen = criterion(d_fake, torch.ones_like(d_fake).to(self.device))
                gen_opt.zero_grad()
                loss_gen.backward()
                gen_opt.step()
                self.gen_history.append(loss_gen.item())

                #train discriminator
                with torch.no_grad():
                    y_fake = self.generator(x_cat, x_cont).detach()
                d_fake = self.discriminator(x_cat, x_cont, y_fake)
                loss_fake = criterion(d_fake, torch.zeros_like(d_fake).to(self.device))
                d_real = self.discriminator(x_cat, x_cont, y)
                loss_real = criterion(d_real.to(self.device), torch.ones_like(d_real).to(self.device))
                loss_dis = (loss_fake+loss_real)/2
                dis_opt.zero_grad()
                loss_dis.backward()
                dis_opt.step()
                self.dis_history.append(loss_dis.item())

        self = self.to('cpu')