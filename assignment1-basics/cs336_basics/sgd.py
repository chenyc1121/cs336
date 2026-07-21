import torch
from collections.abc import Callable,Iterable
from typing import Optional
import math


class SGD(torch.optim.Optimizer):
    def __init__(self,param,lr=1e-3):
        defaults={"lr":lr}
        super().__init__(params=param,defaults=defaults)

    def step(self,closure:Optional[Callable]=None):
        loss =None if closure is None else closure()
        for group in self.param_groups:
            lr=group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data-=lr*p.grad.data
        return loss
                