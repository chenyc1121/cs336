from typing import Any

import torch
import math

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model,eps) -> None:
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=torch.nn.Parameter(torch.ones(d_model))


    def forward(self,x:torch.Tensor):
        in_type=x.dtype
        x=x.to(dtype=torch.float32)
        div=torch.sqrt(torch.mean(x.square(),dim=-1,keepdim=True)+self.eps)
        result=(x/div) * self.weight
        return result.to(in_type)
        