import torch.utils.data as torchdata
import torch

from torch.utils.data import DataLoader

class WassersteinGPCEmbeddingsGANTrainer():
    def __init__(self, model, dataset, seq_length, test_data_coefficient, batch_size, device, embeddings, learning_rate=1e-3, critic_iterations=5, gp_scale=10):
        self.generator = model.generator
        self.critic = model.discriminator
        self.seq_length = seq_length
        self.test_data_coefficient = test_data_coefficient
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.critic_iterations = critic_iterations
        self.gp_scale = gp_scale
        self.summary = "Wasserstein-GPC-" + str(batch_size) + "-" + str(learning_rate) + "-" + str(critic_iterations) + "-" + str(gp_scale)

        testdata_size  = int(len(dataset) * test_data_coefficient)
        train_dataset = torchdata.Subset(dataset=dataset, indices=torch.arange(testdata_size, len(dataset)))
        test_dataset = torchdata.Subset(dataset=dataset, indices=torch.arange(0, testdata_size))
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
        cat, cont, self.test_data_local = next(iter(test_dataloader))
        self.test_cat_local = cat
        self.test_cont_local = cont
        self.test_data_cuda = self.test_data_local.to(device)
        self.test_cat_cuda = self.test_cat_local.to(device)
        self.test_cont_cuda = self.test_cont_local.to(device)

        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.embeddings = embeddings
        print(cat.shape)
        if(len(embeddings) != cat.shape[2]):
            raise ValueError(f"Not enough Embeddings for the number of categorical features! {len(embeddings)} != {cat.shape[2]}")
        self.optimizer_embeddings = [torch.optim.Adam(embedding.parameters(),lr=learning_rate) for embedding in self.embeddings]
                

    def train(self):
        generator_loss_sum = 0
        critic_loss_sum = 0
        for i, inputs in enumerate(self.train_dataloader):
            cat, cont, target = inputs
            cond = cont.to(self.device)
            cat = cat.to(self.device)
            for j in range(len(self.embeddings)):
                cond = torch.cat([cond, self.embeddings[j](cat[:, :, j])], dim=2)
            target = target.to(self.device)

            for _ in range(self.critic_iterations):
                fake = self.generator.getSample(conditions=cond.detach(), n=self.batch_size, length=self.seq_length)
                epsilon = torch.rand(self.batch_size, 1).to(self.device)
                interpolated = epsilon * target + (1 - epsilon) * fake
                critic_interpol = self.critic(interpolated, cond.detach())
                grad_outputs = torch.ones_like(critic_interpol).to(self.device)
                gradients = torch.autograd.grad(outputs=critic_interpol, inputs=interpolated, grad_outputs=grad_outputs, create_graph=True)[0]
                norm = gradients.norm(2,1)
                gradient_penalty = torch.mean((norm - 1) ** 2)
                critic_real = self.critic(target, cond.detach())
                critic_fake = self.critic(fake, cond.detach())
                critic_loss = - torch.mean(critic_real) + torch.mean(critic_fake) + self.gp_scale * gradient_penalty
                critic_loss_sum += critic_loss
                self.critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            fake = self.generator.getSample(conditions=cond, n=self.batch_size, length=self.seq_length)
            fake = fake.view(fake.size(0), fake.size(1), 1)
            output = self.critic(fake, cond)
            generator_loss = - torch.mean(output)
            generator_loss_sum += generator_loss
            self.generator.zero_grad()
            generator_loss.backward()
            self.optimizer_generator.step()
            for optimizer in self.optimizer_embeddings:
                optimizer.step()
                optimizer.zero_grad()

        return {"generator_loss":generator_loss_sum.detach().cpu() / i, "discriminator_loss":critic_loss_sum.detach().cpu() / (i * self.critic_iterations)}