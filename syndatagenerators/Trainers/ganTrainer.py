import torch.utils.data as torchdata
import torch
import numpy as np

from torch.utils.data import DataLoader

class WassersteinGPGANTrainer():#Doesn't work with RNNs cause create_graph=True in grad() :(
    def __init__(self, model, dataset, seq_length, test_data_coefficient, batch_size, device, learning_rate=1e-3, critic_iterations=5, gp_scale=10, val_data_coefficient=0.0):
        self.generator = model.generator
        self.critic = model.discriminator
        self.seq_length = seq_length
        self.test_data_coefficient = test_data_coefficient
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.critic_iterations = critic_iterations
        self.gp_scale = gp_scale
        self.summary = "Wasserstein-GP-" + str(batch_size) + "-" + str(learning_rate) + "-" + str(critic_iterations) + "-" + str(gp_scale)

        testdata_size  = int(len(dataset) * test_data_coefficient)
        val_data_size = int(len(dataset) * val_data_coefficient)

        train_idx = np.arange(len(dataset))
        test_idx = np.random.RandomState(1).choice(train_idx, int(testdata_size), replace=False)
        train_idx = np.delete(train_idx, test_idx)
        val_idx = np.random.RandomState(1).choice(train_idx, int(val_data_size), replace=False)
        train_idx = np.array([i for i in train_idx if i not in val_idx])

        train_dataset = torchdata.Subset(dataset=dataset, indices=train_idx)
        test_dataset = torchdata.Subset(dataset=dataset, indices=test_idx)
        val_dataset = torchdata.Subset(dataset=dataset, indices=val_idx)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=True)
        self.test_data_local = next(iter(test_dataloader)).cpu()
        self.test_data_cuda = self.test_data_local.to(device)

        if(len(val_dataset) > 0):
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=True, drop_last=True)
            self.val_data_local = next(iter(val_dataloader)).cpu()
            self.val_data_cuda = self.val_data_local.to(device)

        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)

    def train(self):
        generator_loss_sum = 0
        critic_loss_sum = 0
        for i, inputs in enumerate(self.train_dataloader):
            real = inputs.to(self.device)

            for _ in range(self.critic_iterations):
                fake = self.generator.getSample(n=self.batch_size, length=self.seq_length)
                epsilon = torch.rand(self.batch_size, 1).to(self.device)
                interpolated = epsilon * real + (1 - epsilon) * fake
                critic_interpol = self.critic(interpolated)
                grad_outputs = torch.ones_like(critic_interpol).to(self.device)
                gradients = torch.autograd.grad(outputs=critic_interpol, inputs=interpolated, grad_outputs=grad_outputs, create_graph=True)[0]
                norm = gradients.norm(2,1)
                gradient_penalty = torch.mean((norm - 1) ** 2)
                critic_real = self.critic(real)
                critic_fake = self.critic(fake)
                critic_loss = - torch.mean(critic_real) + torch.mean(critic_fake) + self.gp_scale * gradient_penalty
                critic_loss_sum += critic_loss
                self.critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            fake = self.generator.getSample(n=self.batch_size, length=self.seq_length)
            output = self.critic(fake)
            generator_loss =  - torch.mean(output)
            generator_loss_sum += generator_loss
            self.generator.zero_grad()
            generator_loss.backward()
            self.optimizer_generator.step()

        return {"generator_loss":generator_loss_sum.detach().cpu() / i, "discriminator_loss":critic_loss_sum.detach().cpu() / (i * self.critic_iterations)}

