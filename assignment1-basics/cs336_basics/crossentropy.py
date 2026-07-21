import torch
from .softmax import softmax

def crossentropy(logit:torch.Tensor,x:torch.Tensor):
    max_logits=logit.max(dim=-1,keepdim=True).values
    shifted_logits=logit-max_logits
    log_partition=torch.log(torch.exp(shifted_logits).sum(dim=-1))
    target_logit=torch.gather(shifted_logits,-1,x.unsqueeze(-1)).squeeze(-1)
    return (log_partition-target_logit).mean()

    