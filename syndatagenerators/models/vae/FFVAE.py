import torch, os, time

class FFVAE(torch.nn.Module):

    def __init__(self, latent_dim, seq_length, num_hiddenunits, num_hiddenlayers, device) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_hiddenunits = num_hiddenunits
        self.num_hiddenlayers = num_hiddenlayers
        self.device = device

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=seq_length, out_features=num_hiddenunits),
            torch.nn.LeakyReLU(),
        )

        for _ in range(num_hiddenlayers):
            self.encoder.append(torch.nn.Linear(in_features=num_hiddenunits, out_features=num_hiddenunits))
            self.encoder.append(torch.nn.LeakyReLU())
        
        self.encoder.append(torch.nn.Linear(in_features=num_hiddenunits, out_features=2*latent_dim))

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_dim, out_features=num_hiddenunits),
            torch.nn.LeakyReLU()
        )

        for _ in range(num_hiddenlayers):
            self.decoder.append(torch.nn.Linear(in_features=num_hiddenunits, out_features=num_hiddenunits))
            self.decoder.append(torch.nn.LeakyReLU())
        
        self.decoder.append(torch.nn.Linear(in_features=num_hiddenunits, out_features=seq_length))

        self.to(device=device)
       
    def getLatentEncoding(self, x):
        enc_out = self.encoder(x).squeeze()
        mu = enc_out[:,self.latent_dim:]
        log_var = enc_out[:,:self.latent_dim]
        std = torch.exp(0.5*log_var)
        noise = torch.randn(mu.shape, device=self.device)
        # sample latent via reparametrization trick
        self.kl_loss = .5*(std**2 + mu**2 - 1 - log_var).sum()
        return mu + std * noise
        
    def decode(self, z) -> torch.Tensor:
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
        additional_content["num_hiddenunits"] = self.num_hiddenunits
        additional_content["num_hiddenlayers"] = self.num_hiddenlayers
        additional_content["model_state_dict"] = self.state_dict()
        torch.save(additional_content, path)
        return path

    def loadModel(path, device):
        load = torch.load(path)
        try:
            model = FFVAE(load["latent_dim"], load["seq_length"], load["num_hiddenunits"], load["num_hiddenlayers"], device)
            model.load_state_dict(load["model_state_dict"])
        except KeyError:
            print("File does not contain the right keys")
            return None
        return model