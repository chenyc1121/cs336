import math

import torch
class embedding(torch.nn.Module):
    def __init__(self,num_embeddings,embedding_dim) -> None:
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.weight=torch.nn.Parameter(torch.empty(num_embeddings,embedding_dim))
        sigma=math.sqrt(2/(num_embeddings+embedding_dim))
        torch.nn.init.trunc_normal_(self.weight,std=sigma,a=-3*sigma,b=3*sigma)
    
    def forward(self,x):
        return self.weight[x]