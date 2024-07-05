import torch.utils.data as torchdata
import torch
import numpy as np

from torch.utils.data import DataLoader
class BasicVAETrainer():
    def __init__(self, model, dataset, seq_length, test_data_coefficient, batch_size, device, kl_weight, lr, val_data_coefficient=0.0):
        self.model = model
        self.seq_length = seq_length
        self.test_data_coefficient = test_data_coefficient
        self.device = device
        self.kl_weight = kl_weight
        self.summary = str(batch_size) + "-" + str(kl_weight) + "-" + str(lr)

        testdata_size  = int(len(dataset) * test_data_coefficient)
        val_data_size = int(len(dataset) * val_data_coefficient)

        train_idx = np.arange(len(dataset))
        test_idx = np.random.RandomState(1).choice(train_idx, int(testdata_size), replace=False)
        train_idx = np.delete(train_idx, test_idx)
        val_idx = np.random.RandomState(2).choice(train_idx, int(val_data_size), replace=False)
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

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        

    def train(self) -> dict:
        mse_loss_sum = 0
        kl_loss_sum = 0
        for i, inputs in enumerate(self.train_dataloader):
            x = inputs.to(self.device)
            x_recon =  self.model(x)
            mse_loss = self.criterion(x_recon,x)
            loss = mse_loss + self.kl_weight * self.model.kl_loss
            mse_loss_sum += mse_loss
            kl_loss_sum += self.model.kl_loss
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {"mse_loss":mse_loss_sum.detach().cpu() / i, "kl_loss":kl_loss_sum.detach().cpu() / i}