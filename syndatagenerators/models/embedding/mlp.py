import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm import tqdm

class BasicMLP(nn.Module):
    def __init__(self, n_features, n_hidden, n_targets):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_targets = n_targets

        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_targets),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class BayesianMtlEmbeddingLearner(nn.Module):
    def __init__(self, model, n_tasks, embedding_dim=2, kl_weight=1e-3, device=torch.device('cuda')):
        super().__init__()
        self.model = model
        self.n_tasks = n_tasks
        self.embedding_dim = embedding_dim
        self.kl_weight = kl_weight
        self.device = device
        self.mu_embed = nn.Embedding(n_tasks, embedding_dim)
        self.log_var_embed = nn.Embedding(n_tasks, embedding_dim)
        self.kl_loss = 0.0
        self.criterion_history = []
        self.kl_history = []

        self.net = nn.Sequential(
            nn.Linear(embedding_dim+self.model.n_features, self.model.n_features),
            nn.LeakyReLU(),
            self.model()
        )

    def init_embeddings(self):
        self.mu_embed.weight.data = torch.randn_like(self.mu_embed.weight.data)
        self.log_var_embed.weight.data = torch.zeros_like(self.log_var_embed.weight.data) - 1.
  
    def forward(self, x_cat, x_cont, train_flag=True):
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
        x_cont = torch.cat([x_cont, e], dim=2)
        y = self.net(x_cont)
        return y

    def train(self, dl_train, optimizer, criterion, n_epochs):
        self = self.to(self.device)
        for epoch in tqdm(range(n_epochs)):
            for x_cat, x_cont, y in dl_train:
                x_cat = x_cat.to(self.device)
                x_cont = x_cont.to(self.device)
                y = y.to(self.device)
                # forward pass
                pred = self.forward(x_cat, x_cont, train_flag=True)
                loss = criterion(pred, y) + self.kl_weight*self.kl_loss
                self.criterion_history.append(loss.item())
                self.kl_history.append(self.kl_loss.item())
                # backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        self = self.to('cpu')


    def get_test_samples(self, dl_test):
        x_cat = []
        x_cont = []
        y = []

        self = self.to(self.device)
        for x_cat_b, x_cont_b, y_b in dl_test:
            x_cat.append(x_cat_b.to(self.device))
            x_cont.append(x_cont_b.to(self.device))
            y.append(y_b.to(self.device))

        x_cat, x_cont, y = torch.cat(x_cat), torch.cat(x_cont), torch.cat(y)
        pred = self.forward(x_cat, x_cont, train_flag=False)

        test_samples = {}
        test_samples['x_cat'] = x_cat
        test_samples['x_cont'] = x_cont
        test_samples['y'] = y
        test_samples['pred'] = pred

        return test_samples

    def plot_bayes_embedding_mu_sigma(self):
        E = self.mu_embed.weight.data
        std = torch.exp(0.5*self.log_var_embed.weight.data)

        for task in range(self.n_tasks):
            plt.subplot(self.n_tasks, 1, task + 1)
            plt.plot(E[task],'sk')
            plt.plot(E[task] + std[task],'+')
            plt.plot(E[task] - std[task],'+')

    def plot_bayes_embedding_space(self, show_circles=False):
        E = self.mu_embed.weight.data
        std = torch.exp(0.5*self.log_var_embed.weight.data)

        sort_idx = torch.argsort(torch.mean(std,dim=0))
        first = sort_idx[0]
        second = sort_idx[1]

        tt = torch.linspace(0, 2*torch.pi, 100)
        if show_circles:
            for e,s in zip(E, std):
                plt.plot(e[first] + s[first]*torch.sin(tt), e[second] + s[second]*torch.cos(tt))
        
        plt.scatter(E[:,first], E[:,second])
        plt.plot(torch.sin(tt), torch.cos(tt), 'k--')
