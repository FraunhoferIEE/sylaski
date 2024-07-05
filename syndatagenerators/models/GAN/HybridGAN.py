import torch, time, os
class ManyToManyHybridLSTMFFWGAN():

    class Generator(torch.nn.Module):

        def __init__(self, num_hiddenunits, num_hiddenlayers, num_features, device) -> None:
            super().__init__()
            self.num_features = num_features
            self.device = device
            self.inputLSTM = torch.nn.LSTM(input_size=num_features, hidden_size=num_hiddenunits, num_layers=num_hiddenlayers+1, batch_first=True)
            self.outputLSTM = torch.nn.LSTM(input_size=num_hiddenunits, hidden_size=1, num_layers=1, batch_first=True)

        def forward(self, x):
            out, (_, _) = self.inputLSTM(x)
            out, (_, _) = self.outputLSTM(out)
            return out.squeeze()

        def getSample(self, n, length):
            z = torch.randn(n, length, self.num_features, device=self.device)
            out, (_, _) = self.inputLSTM(z)
            out, (_, _) = self.outputLSTM(out)
            return out.squeeze()

        def generate(self, z):
            out, (_, _) = self.inputLSTM(z)
            out, (_, _) = self.outputLSTM(out)
            return out.squeeze()

    class Discriminator(torch.nn.Module):

        def __init__(self, num_hiddenunits, num_hiddenlayers, seq_length) -> None:
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

    def __init__(self, num_hiddenunits, num_hiddenlayers, num_features, device, seq_length) -> None:
        self.num_hiddenunits = num_hiddenunits
        self.num_hiddenlayers = num_hiddenlayers
        self.num_features = num_features
        self.seq_length = seq_length

        self.generator = self.Generator(num_hiddenunits=num_hiddenunits, num_hiddenlayers=num_hiddenlayers, num_features=num_features, device=device).to(device)
        self.discriminator = self.Discriminator(num_hiddenunits=num_hiddenunits, num_hiddenlayers=num_hiddenlayers, seq_length=seq_length).to(device)

    def getLatentVector(self, n, length, device="cpu"):
        return torch.randn(n, length, self.num_features, device=device)

    def saveModel(self, path:str="", additional_content:dict=None):
        if(additional_content is None):
            additional_content = {}
        if(path != "" and not os.path.isdir(path)):
            os.mkdir(path)
        path = path+self.__class__.__name__+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())+".pt"
        additional_content["num_hiddenunits"] = self.num_hiddenunits
        additional_content["num_hiddenlayers"] = self.num_hiddenlayers
        additional_content["num_features"] = self.num_features
        additional_content["seq_length"] = self.seq_length
        additional_content["generator_state_dict"] = self.generator.state_dict()
        additional_content["discriminator_state_dict"] = self.discriminator.state_dict()
        torch.save(additional_content, path)
        return path

    def loadModel(path, device):
        load = torch.load(path)
        try:
            model = ManyToManyHybridLSTMFFWGAN(load["num_hiddenunits"], load["num_hiddenlayers"], load["num_features"], device, load["seq_length"])
            model.generator.load_state_dict(load["generator_state_dict"])
            model.discriminator.load_state_dict(load["discriminator_state_dict"])
        except KeyError:
            print("File does not contain the right keys")
            return None
        return model