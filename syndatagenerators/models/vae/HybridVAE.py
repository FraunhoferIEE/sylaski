import torch, os, time

class OneToManyHybridLSTMFFVAE(torch.nn.Module):

    def __init__(self, latent_dim, num_hiddenunits, seq_length, num_hiddenlayers, device) -> None:
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_hiddenunits = num_hiddenunits
        self.seq_length = seq_length
        self.num_hiddenlayers = num_hiddenlayers

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=seq_length, out_features=num_hiddenunits),
            torch.nn.LeakyReLU(),
        )

        for _ in range(num_hiddenlayers):
            self.encoder.append(torch.nn.Linear(in_features=num_hiddenunits, out_features=num_hiddenunits))
            self.encoder.append(torch.nn.LeakyReLU())
        
        self.encoder.append(torch.nn.Linear(in_features=num_hiddenunits, out_features=2*latent_dim))

        self.decode_inputlstm = torch.nn.LSTM(input_size=latent_dim, hidden_size=num_hiddenunits, num_layers=num_hiddenlayers+1, batch_first=True)
        self.decode_outputlstm = torch.nn.LSTM(input_size=num_hiddenunits, hidden_size=1, num_layers=1, batch_first=True)


        self.to(device=device)
       
    def getLatentEncoding(self, x):
        enc_out = self.encoder(x)
        mu = enc_out[:, self.latent_dim:]
        log_var = enc_out[:, :self.latent_dim]
        std = torch.exp(0.5*log_var)
        # sample latent via reparametrization trick
        self.kl_loss = .5*(std**2 + mu**2 - 1 - log_var).sum()
        noise = torch.randn(mu.shape, device=self.device)
        latent = torch.zeros((x.size(0), x.size(1), self.latent_dim), device=self.device)
        latent[:, 0, :] = mu + std * noise
        return latent
        
    def decode(self, z):
        out, (_, _) = self.decode_inputlstm(z)
        out, (_, _) = self.decode_outputlstm(out)
        return out.squeeze()

    def forward(self, x):
        return self.decode(self.getLatentEncoding(x))

    def getSample(self, n, length):
        return self.decode(self.getLatentVector(n, length, self.device))
    
    def getLatentVector(self, n, length, device):
        latent_vector = torch.zeros(n, length, self.latent_dim)
        for i in range(n):  
            latent_vector[i,0] = torch.randn(self.latent_dim)
        return latent_vector.to(device)
    
    def saveModel(self, path:str="", additional_content:dict=None):
        if(additional_content is None):
            additional_content = {}
        if(path != "" and not os.path.isdir(path)):
            os.mkdir(path)
        path = path+self.__class__.__name__+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())+".pt"
        additional_content["latent_dim"] = self.latent_dim
        additional_content["num_hiddenunits"] = self.num_hiddenunits
        additional_content["seq_length"] = self.seq_length
        additional_content["num_hiddenlayers"] = self.num_hiddenlayers
        additional_content["model_state_dict"] = self.state_dict()
        torch.save(additional_content, path)
        return path

    def loadModel(path, device):
        load = torch.load(path)
        try:
            model = OneToManyHybridLSTMFFVAE(load["latent_dim"], load["num_hiddenunits"], load["seq_length"], load["num_hiddenlayers"], device)
            model.load_state_dict(load["model_state_dict"])
        except KeyError:
            print("File does not contain the right keys")
            return None
        return model