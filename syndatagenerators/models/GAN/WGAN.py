import torch, time, os

class WGAN():

    class Generator(torch.nn.Module):
        def __init__(self, seq_length, latent_dim, num_hiddenunits, num_hiddenlayers, device):
            super().__init__()
            self.latent_dim = latent_dim
            self.device = device
            self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim , num_hiddenunits),
                    torch.nn.LeakyReLU(),
                )
            
            for _ in range(num_hiddenlayers):
                self.net.append(torch.nn.Linear(num_hiddenunits,  num_hiddenunits))
                self.net.append(torch.nn.LeakyReLU())
            
            self.net.append(torch.nn.Linear(num_hiddenunits,  seq_length))

        def forward(self, x):
            return self.net(x)

        def getSample(self, n, length):
            return self.forward(self.getLatentVector(n, length, self.device))
        
        def getLatentVector(self, n, length, device):
            return torch.randn(n, self.latent_dim, device=device)
        
    class Discriminator(torch.nn.Module):
        def __init__(self, seq_length, num_hiddenunits, num_hiddenlayers):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(seq_length, num_hiddenunits),
                torch.nn.LeakyReLU(),
            )
            for _ in range(num_hiddenlayers):
                self.net.append(torch.nn.Linear(num_hiddenunits, num_hiddenunits))
                self.net.append(torch.nn.LeakyReLU())

            self.net.append(torch.nn.Linear(num_hiddenunits, 1))

        def forward(self, x):
            return self.net(x)

    def __init__(self, num_hiddenunits, latent_dim, num_hiddenlayers, seq_length, device) -> None:
        self.num_hiddenunits = num_hiddenunits
        self.latent_dim = latent_dim
        self.num_hiddenlayers = num_hiddenlayers
        self.seq_length = seq_length

        self.generator = self.Generator(seq_length=seq_length, latent_dim=latent_dim, num_hiddenunits=num_hiddenunits, num_hiddenlayers=num_hiddenlayers, device=device).to(device)
        self.discriminator = self.Discriminator(seq_length=seq_length, num_hiddenunits=num_hiddenunits, num_hiddenlayers=num_hiddenlayers).to(device)

    def getLatentVector(self, n, length, device):
        return self.generator.getLatentVector(n, length, device)

    def saveModel(self, path:str="", additional_content:dict=None):
        if(additional_content is None):
            additional_content = {}
        if(path != "" and not os.path.isdir(path)):
            os.mkdir(path)
        path = path+self.__class__.__name__+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())+".pt"
        additional_content["num_hiddenunits"] = self.num_hiddenunits
        additional_content["num_hiddenlayers"] = self.num_hiddenlayers
        additional_content["latent_dim"] = self.latent_dim
        additional_content["seq_length"] = self.seq_length
        additional_content["generator_state_dict"] = self.generator.state_dict()
        additional_content["discriminator_state_dict"] = self.discriminator.state_dict()
        torch.save(additional_content, path)
        return path

    def loadModel(path, device):
        load = torch.load(path)
        try:
            model = WGAN(load["num_hiddenunits"], load["latent_dim"], load["num_hiddenlayers"], load["seq_length"], device)
            model.generator.load_state_dict(load["generator_state_dict"])
            model.discriminator.load_state_dict(load["discriminator_state_dict"])
        except KeyError:
            print("File does not contain the right keys")
            return None
        return model    
