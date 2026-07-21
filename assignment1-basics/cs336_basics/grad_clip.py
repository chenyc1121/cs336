import torch
from collections.abc import Iterable

def grad_clip(params:Iterable[torch.nn.Parameter],max_l2,eps=1e-6):
    grads: list[torch.Tensor] = []
    for param in params:
        grad = param.grad
        if grad is not None:
            grads.append(grad)

    if not grads:
        return

    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(grad.detach()) for grad in grads])
    )
    clip_coef = max_l2 / (total_norm + eps)
    if clip_coef < 1:
        with torch.no_grad():
            for grad in grads:
                grad.mul_(clip_coef)
            
