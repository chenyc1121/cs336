from .linear import Linear
from .silu import Silu
import torch

class FFN(torch.nn.Module):
    def __init__(self,d_model,d_ff: int | None = None):
        super().__init__()
        self.dim_in=d_model
        if d_ff is None:
            self.d_ff=((d_model*8//3-32)//64+1)*64
        else :
            self.d_ff=d_ff
        self.w1=Linear(self.dim_in,self.d_ff)
        self.w3=Linear(self.dim_in,self.d_ff)
        self.w2=Linear(self.d_ff,self.dim_in)
    
    def forward(self,x:torch.Tensor):
        return self.w2(Silu()(self.w1(x))*self.w3(x))
