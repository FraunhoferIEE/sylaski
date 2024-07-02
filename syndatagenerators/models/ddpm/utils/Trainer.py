from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
from syndatagenerators.models.ddpm.utils.DiffusionUtilities import DiffusionUtilities
from syndatagenerators.models.ddpm.architectures.UNetPredictor2 import EMA
from torch.utils.data import DataLoader
import pandas as pd
from syndatagenerators.metrics.mmd_score import new_mmd

class Trainer():
    def __init__(self, 
                 ddpm: DiffusionUtilities, 
                 loader: DataLoader ,
                 n_epochs: int,
                 batch_size:int,
                 train_size:int,
                 lr: float,
                 store_path: str,
                 tag: str,
                 household: str, 
                 device: str ="cpu"):
        self.ddpm = ddpm
        self.tag = tag
        self.household = household
        self.n_epochs = n_epochs
        self.batch_size= batch_size
        self.train_size = train_size
        self.lr = lr
        self.loader = loader
        self.device = device
        self.store_path = store_path
        self.optimizer = torch.optim.AdamW(self.ddpm.network.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.n_epochs)
        self.criterion = torch.nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def saveModel(self, model, path):
        """Saves state of an model for future loading

        Args:
            model (Neural Network): The model that is saved
            path (String): Location where to save it
        """
        torch.save(model.state_dict(), path)
    


    def train_ddpm(self):
        losses = []
        mean_losses = []
        epoch_loss = 1.0
        self.ddpm.network = self.ddpm.network.to(self.device)
        for epoch in tqdm(range(self.n_epochs), desc="Training...", colour="#ffffff"):
            for step, batch in enumerate(tqdm(self.loader, leave=False, desc=f"epoch: {epoch + 1}/{self.n_epochs} last loss: {epoch_loss}", colour="#55ff55")):
                lable,cont,x = batch
                x = x.squeeze()
                x0 = x.to(self.device)

                if self.ddpm.network.num_classes is None:
                    y = None
                else:
                    y = lable[:,0,:].long().to(self.device)

                cont = cont[:,0,:].float().to(self.device)

                noise = torch.randn_like(x0).to(self.device)

                t = torch.randint(
                    0, self.ddpm.n_steps, (x0.shape[0], 1), dtype=torch.int64).to(self.device)
                
                noisy_images = self.ddpm.addNoise(x0, t, noise)
                noisy_images = noisy_images.view(-1,1,self.train_size).to(self.device)
                
                predNoise = self.ddpm.predictNoise(noisy_images, t.squeeze(),cat=y,cont=cont)

                mse = self.criterion(predNoise[:,0,:],noise)
                mmd = abs(new_mmd(predNoise[:,0,:],noise,self.device))

                loss = mse + mmd 

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.step_ema(self.ddpm.ema_model, self.ddpm.network)
                self.scheduler.step()
                losses.append(loss.item())

            if(loss.item() < epoch_loss):
                epoch_loss = loss.item()
                self.saveModel(self.ddpm.network, f'{self.store_path}network.pt')
                self.saveModel(self.ddpm.network.task_emb, f'{self.store_path}embedding.pt')

            mean_losses.append(sum(losses)/len(losses))
            losses = []
            plt.semilogy(mean_losses)
            plt.xlabel("epoch")
            plt.title(f"{self.household} loss")
            plt.savefig(
                f"{self.store_path}loss.png")
            plt.close()

       