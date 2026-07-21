from typing import Any

import torch

class softmax(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x:torch.Tensor,dim=-1):
        x_max=torch.max(x,dim=dim,keepdim=True).values
        exp=torch.exp(x-x_max)
        div=torch.sum(exp,dim=dim,keepdim=True)
        return exp/div