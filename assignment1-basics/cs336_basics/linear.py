import torch
import math
class Linear(torch.nn.Module):
    def __init__(self, in_features,out_features):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        sigma = math.sqrt(2 / (in_features + out_features))
        w=torch.empty(out_features,in_features)
        self.weight=torch.nn.Parameter(w)
        torch.nn.init.trunc_normal_(self.weight,std=sigma,a=-3*sigma,b=3*sigma)
    
    def forward(self,x:torch.Tensor):
        return x @ self.weight.T
