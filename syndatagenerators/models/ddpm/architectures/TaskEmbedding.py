import torch
import torch.nn as nn

class TaskEmbedding(nn.Module):
    """Embed a single categorical predictor
    
    Keyword Arguments:
    
    num_output_classes: int, number of output classes
    num_cat_classes: list[int], number of classes for each categorical variable
    num_cont: int, number of continuous variables
    embedding_dim: int, dimension of the embedding
    hidden_dim: int, dimension of the hidden layer
    """
    def __init__(self, num_output_classes:int, num_cat_classes:list[int], num_cont:int, embedding_dim:int=64, hidden_dim:int=64):
        super().__init__()
        # Create an embedding for each categorical input
        self.embeddings = nn.ModuleList([nn.Embedding(nc, embedding_dim) for nc in num_cat_classes])
        self.fc1 = nn.Linear(in_features=len(num_cat_classes) * embedding_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=num_cont, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(2*hidden_dim, num_output_classes)
        
    def forward(self, x_cat, x_con):
        # Embed each of the categorical variables

        x_embed = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(x_embed, dim=1)
        x = self.fc1(x_embed)
        x_con = self.fc2(x_con)
        x = torch.cat([x_con, x], dim=1)
        x = self.relu(x)
        return self.out(x)    