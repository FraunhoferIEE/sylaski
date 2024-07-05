import torch, time, os

class DCWGAN():

    class Generator(torch.nn.Module):
        def __init__(self, latent_dim, seq_length, num_hiddenchannels, num_hiddenlayers, device) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.device = device
            kernel_size = seq_length // (num_hiddenlayers + 2) + 1
            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(in_channels=latent_dim, out_channels=num_hiddenchannels, kernel_size=kernel_size),
                torch.nn.LeakyReLU(),
            )

            for _ in range(num_hiddenlayers):
                self.net.append(torch.nn.ConvTranspose1d(in_channels=num_hiddenchannels, out_channels=num_hiddenchannels, kernel_size=kernel_size))
                self.net.append(torch.nn.LeakyReLU())
            
            self.net.append(torch.nn.ConvTranspose1d(in_channels=num_hiddenchannels, out_channels=1, kernel_size=seq_length - (num_hiddenlayers + 1) * (kernel_size - 1)))

        def forward(self, x):
            x = x.view(x.size(0), x.size(1), 1)
            return self.net(x).squeeze()
            
        def getSample(self, n, length):
            return self.forward(self.getLatentVector(n, length, self.device))

        def getLatentVector(self, n, length, device):
            return torch.randn(n, self.latent_dim, device=device)

    class Discriminator(torch.nn.Module):
        def __init__(self, seq_length, num_hiddenchannels, num_hiddenlayers) -> None:
            super().__init__()
            kernel_size = seq_length // (num_hiddenlayers + 2) + 1
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=1, out_channels=num_hiddenchannels, kernel_size=kernel_size),
                torch.nn.LeakyReLU(),
            )

            for _ in range(num_hiddenlayers):
                self.net.append(torch.nn.Conv1d(in_channels=num_hiddenchannels, out_channels=num_hiddenchannels, kernel_size=kernel_size))
                self.net.append(torch.nn.LeakyReLU())
            
            self.net.append(torch.nn.Conv1d(in_channels=num_hiddenchannels, out_channels=1, kernel_size=seq_length - (num_hiddenlayers + 1) * (kernel_size - 1)))

        def forward(self, x):
            x = x.view(x.size(0), 1, x.size(1))
            return self.net(x).squeeze()

    def __init__(self, seq_length, latent_dim, num_hiddenchannels, num_hiddenlayers, device):
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.num_hiddenchannels = num_hiddenchannels
        self.num_hiddenlayers = num_hiddenlayers
        
        self.generator = self.Generator(latent_dim, seq_length, num_hiddenchannels, num_hiddenlayers, device).to(device)
        self.discriminator = self.Discriminator(seq_length, num_hiddenchannels, num_hiddenlayers).to(device)

    def getLatentVector(self, n, length, device):
        return self.generator.getLatentVector(n, length, device)

    def saveModel(self, path:str="", additional_content:dict=None):
        if(additional_content is None):
            additional_content = {}
        if(path != "" and not os.path.isdir(path)):
            os.mkdir(path)
        path = path+self.__class__.__name__+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())+".pt"
        additional_content["seq_length"] = self.seq_length
        additional_content["latent_dim"] = self.latent_dim
        additional_content["num_hiddenchannels"] = self.num_hiddenchannels
        additional_content["num_hiddenlayers"] = self.num_hiddenlayers
        additional_content["generator_state_dict"] = self.generator.state_dict()
        additional_content["discriminator_state_dict"] = self.discriminator.state_dict()
        torch.save(additional_content, path)
        return path

    def loadModel(path, device):
        load = torch.load(path)
        try:
            model = DCWGAN(load["seq_length"], load["latent_dim"], load["num_hiddenchannels"], load["num_hiddenlayers"], device)
            model.generator.load_state_dict(load["generator_state_dict"])
            model.discriminator.load_state_dict(load["discriminator_state_dict"])
        except KeyError:
            print("File does not contain the right keys")
            return None
        return model