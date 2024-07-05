import torch, time, os

class BasicEmbeddingCManyToManyHybridLSTMWFFGAN():

    class Generator(torch.nn.Module):

        def __init__(self, num_hiddenunits, num_hiddenlayers, num_features, lens_cat_conditions, num_cont, embedding_dim, embedding_out, device) -> None:
            super().__init__()
            self.num_features = num_features
            self.device = device
            self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(len_cat_condition, embedding_dim, device=device) for len_cat_condition in lens_cat_conditions])
            self.embedding_net = torch.nn.Sequential(
                torch.nn.Linear(len(lens_cat_conditions) * embedding_dim + num_cont, embedding_out),
                torch.nn.LeakyReLU(),
            )
            self.inputLSTM = torch.nn.LSTM(input_size=num_features + embedding_out, hidden_size=num_hiddenunits, num_layers=num_hiddenlayers+1, batch_first=True)
            self.outputLSTM = torch.nn.LSTM(input_size=num_hiddenunits, hidden_size=1, num_layers=1, batch_first=True)

        def forward(self, x, cat, cont):
            embedding = self.getEmbedding(cat, cont)
            input = torch.cat([x, embedding], 2)
            out, (_, _) = self.inputLSTM(input)
            out, (_, _) = self.outputLSTM(out)
            return out.squeeze()

        def getEmbedding(self, cat, cont):
            cond = cont
            for j in range(len(self.embeddings)):
                cond = torch.cat([cond, self.embeddings[j](cat[:, :, j])], dim=2)
            y = self.embedding_net(cond)
            return y


        def getSample(self, n, length, cat, cont):
            return self.forward(self.getLatentVector(n, length, self.device), cat, cont)
        
        def getLatentVector(self, n, length, device):
            return torch.randn(n, length, self.num_features, device=device)

    class Discriminator(torch.nn.Module):

        def __init__(self, num_hiddenunits, num_hiddenlayers, seq_length, lens_cat_conditions, num_cont, embedding_dim, embedding_out, device) -> None:
            super().__init__()

            self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(len_cat_condition, embedding_dim, device=device) for len_cat_condition in lens_cat_conditions])
            self.embedding_net = torch.nn.Sequential(
                torch.nn.Linear(len(lens_cat_conditions) * embedding_dim + num_cont, embedding_out),
                torch.nn.LeakyReLU(),
            )

            self.net = torch.nn.Sequential(
                torch.nn.Linear(seq_length + embedding_out, num_hiddenunits),
                torch.nn.LeakyReLU(),
            )

            for _ in range(num_hiddenlayers):
                self.net.append(torch.nn.Linear(num_hiddenunits, num_hiddenunits))
                self.net.append(torch.nn.LeakyReLU())

            self.net.append(torch.nn.Linear(num_hiddenunits, 1))

        def forward(self, x, cat, cont):
            x = x.squeeze()
            embedding = self.getEmbedding(cat, cont)[:, 0, :]
            input = torch.cat([x, embedding], 1)
            return self.net(input).squeeze()
        
        def getEmbedding(self, cat, cont):
            cond = cont
            for j in range(len(self.embeddings)):
                cond = torch.cat([cond, self.embeddings[j](cat[:, :, j])], dim=2)
            y = self.embedding_net(cond)
            return y

    def __init__(self, num_hiddenunits, num_hiddenlayers, num_features, device, seq_length, lens_cat_conditions, num_cont, embedding_dim, embedding_out) -> None:
        self.num_hiddenunits = num_hiddenunits
        self.num_hiddenlayers = num_hiddenlayers
        self.num_features = num_features
        self.seq_length = seq_length
        self.lens_cat_conditions = lens_cat_conditions
        self.num_cont = num_cont
        self.embedding_dim = embedding_dim
        self.embedding_out = embedding_out

        self.generator = self.Generator(num_hiddenunits=num_hiddenunits, num_hiddenlayers=num_hiddenlayers, num_features=num_features, 
                                        lens_cat_conditions=lens_cat_conditions, num_cont=num_cont, embedding_dim=embedding_dim, embedding_out=embedding_out, device=device).to(device)
        self.discriminator = self.Discriminator(num_hiddenunits=num_hiddenunits, num_hiddenlayers=num_hiddenlayers, seq_length=seq_length,
                                                lens_cat_conditions=lens_cat_conditions, num_cont=num_cont, embedding_dim=embedding_dim, embedding_out=embedding_out, device=device).to(device)

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
        additional_content["num_features"] = self.num_features
        additional_content["seq_length"] = self.seq_length
        additional_content["lens_cat_conditions"] = self.lens_cat_conditions
        additional_content["num_cont"] = self.num_cont
        additional_content["embedding_dim"] = self.embedding_dim
        additional_content["embedding_out"] = self.embedding_out
        additional_content["generator_state_dict"] = self.generator.state_dict()
        additional_content["discriminator_state_dict"] = self.discriminator.state_dict()
        additional_content["generator_embeddings_state_dict"] = [embedding.state_dict() for embedding in self.generator.embeddings]
        additional_content["discriminator_embeddings_state_dict"] = [embedding.state_dict() for embedding in self.discriminator.embeddings]
        torch.save(additional_content, path)
        return path

    def loadModel(path, device):
        load = torch.load(path)
        try:
            model = BasicEmbeddingCManyToManyHybridLSTMWFFGAN(load["num_hiddenunits"], load["num_hiddenlayers"], load["num_features"], device, load["seq_length"], load["lens_cat_conditions"], load["num_cont"], load["embedding_dim"], load["embedding_out"])
            model.generator.load_state_dict(load["generator_state_dict"])
            for e, state in zip(model.generator.embeddings, load["generator_embeddings_state_dict"]):
                e.load_state_dict(state)
            model.discriminator.load_state_dict(load["discriminator_state_dict"])
            for e, state in zip(model.discriminator.embeddings, load["discriminator_embeddings_state_dict"]):
                e.load_state_dict(state)
        except KeyError as e:
            print("File does not contain the right keys")
            print(e)
            return None
        return model