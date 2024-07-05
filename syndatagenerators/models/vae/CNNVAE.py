import torch, time, os

class BasicCNNVAE(torch.nn.Module):# try with samller lr maybe

    def __init__(self, latent_dim, seq_length, num_hiddenchannels, num_hiddenlayers, device) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_hiddenchannels = num_hiddenchannels
        self.num_hiddenlayers = num_hiddenlayers
        self.device = device

        kernel_size = seq_length // (num_hiddenlayers + 2) + 1
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=num_hiddenchannels, kernel_size=kernel_size),
            torch.nn.LeakyReLU(),
        )

        for _ in range(num_hiddenlayers):
            self.encoder.append(torch.nn.Conv1d(in_channels=num_hiddenchannels, out_channels=num_hiddenchannels, kernel_size=kernel_size))
            self.encoder.append(torch.nn.LeakyReLU())
        
        self.encoder.append(torch.nn.Conv1d(in_channels=num_hiddenchannels, out_channels=2*latent_dim, kernel_size=seq_length - (num_hiddenlayers + 1) * (kernel_size - 1)))

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=latent_dim, out_channels=num_hiddenchannels, kernel_size=seq_length - (num_hiddenlayers + 1) * (kernel_size - 1)),
            torch.nn.LeakyReLU()
        )

        for _ in range(num_hiddenlayers):
            self.decoder.append(torch.nn.ConvTranspose1d(in_channels=num_hiddenchannels, out_channels=num_hiddenchannels, kernel_size=kernel_size))
            self.decoder.append(torch.nn.LeakyReLU())
        
        self.decoder.append(torch.nn.ConvTranspose1d(in_channels=num_hiddenchannels, out_channels=1, kernel_size=kernel_size))

        self.to(device=device)
       
    def getLatentEncoding(self, x):
        x = x.view(x.size(0), 1 ,x.size(1))
        enc_out = self.encoder(x).squeeze()
        mu = enc_out[:,self.latent_dim:]
        log_var = enc_out[:,:self.latent_dim]
        std = torch.exp(0.5*log_var)
        noise = torch.randn(mu.shape, device=self.device)
        # sample latent via reparametrization trick
        self.kl_loss = .5*(std**2 + mu**2 - 1 - log_var).sum()
        return mu + std * noise
        
    def decode(self, z) -> torch.Tensor:
        z = z.view(z.size(0), z.size(1), 1)
        return self.decoder(z).squeeze()

    def forward(self, x):
        return self.decode(self.getLatentEncoding(x))

    def getSample(self, n, length):
        return self.decode(self.getLatentVector(n, length, self.device))
    
    def getLatentVector(self, n, length, device):
        return torch.randn(n, self.latent_dim, device=device)

    def saveModel(self, path:str="", additional_content:dict=None):
        if(additional_content is None):
            additional_content = {}
        if(path != "" and not os.path.isdir(path)):
            os.mkdir(path)
        path = path+self.__class__.__name__+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())+".pt"
        additional_content["latent_dim"] = self.latent_dim
        additional_content["seq_length"] = self.seq_length
        additional_content["num_hiddenchannels"] = self.num_hiddenchannels
        additional_content["num_hiddenlayers"] = self.num_hiddenlayers
        additional_content["model_state_dict"] = self.state_dict()
        torch.save(additional_content, path)
        return path

    def loadModel(path, device):
        load = torch.load(path)
        try:
            model = BasicCNNVAE(load["latent_dim"], load["seq_length"], load["num_hiddenchannels"], load["num_hiddenlayers"], device)
            model.load_state_dict(load["model_state_dict"])
        except KeyError:
            print("File does not contain the right keys")
            return None
        return model