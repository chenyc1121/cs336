from .attention import multihead_attention
from .embeding import embedding
from .ffn import FFN
from .linear import Linear
from .rmsnorm import RMSNorm

import torch

class transformer_block(torch.nn.Module):
    def __init__(self,d_model,num_heads,dff,theta,context_len,eps=1e-5):
        super().__init__()
        self.ln1=RMSNorm(d_model,eps)
        self.ln2=RMSNorm(d_model,eps)
        self.attn=multihead_attention(d_model,num_heads,theta,context_len)
        self.ffn=FFN(d_model,dff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x=x+self.attn(self.ln1(x), token_positions)
        x=x+self.ffn(self.ln2(x))
        return x


class transformer(torch.nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta,eps=1e-5):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([transformer_block(d_model,num_heads,d_ff,rope_theta,context_length,eps) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor):
        seq_len = in_indices.shape[-1]
        token_positions = torch.arange(seq_len, device=in_indices.device)
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions)

        return self.lm_head(self.ln_final(x))
