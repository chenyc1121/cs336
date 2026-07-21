import math

import torch
from cs336_basics.softmax import softmax 
from cs336_basics.linear import Linear
from cs336_basics.rope import Rope

def scale_dot_product_attention(Q: torch.Tensor,K: torch.Tensor,V: torch.Tensor,mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(Q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax()(scores, dim=-1)
    return attention_weights @ V

class multihead_attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, theta: float | None = 10000, max_seq_len: int | None = None):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads

        self.q_proj=Linear(d_model,d_model)
        self.k_proj=Linear(d_model,d_model)
        self.v_proj=Linear(d_model,d_model)
        self.output_proj=Linear(d_model,d_model)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope=Rope(theta,self.d_head,max_seq_len)
        
    def split_head(self,x:torch.Tensor):
        x=x.unflatten(-1,(self.num_heads,self.d_head))
        return x.transpose(-3,-2)
    
    def forward(self, in_feature: torch.Tensor, token_positions: torch.Tensor | None = None):
        q=self.split_head(self.q_proj(in_feature))
        k=self.split_head(self.k_proj(in_feature))
        v=self.split_head(self.v_proj(in_feature))

        if self.rope is not None:
            if token_positions is None:
                token_positions=torch.arange(
                    in_feature.shape[-2], device=in_feature.device
                )

            if token_positions.ndim == in_feature.ndim - 1:
                token_positions = token_positions.unsqueeze(-2)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = in_feature.shape[-2]
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=in_feature.device
        ).tril()
        o = scale_dot_product_attention(q, k, v, causal_mask)
        o = o.transpose(-3, -2).flatten(-2)
        return self.output_proj(o)
