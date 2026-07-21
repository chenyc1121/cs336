import torch
import math

class Rope(torch.nn.Module):
    sin_cache: torch.Tensor
    cos_cache: torch.Tensor

    def __init__(self,theta,d_k,max_seq_len):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len

        positions = torch.arange(
            max_seq_len, dtype=torch.float32
        )
        dimensions = torch.arange(
            0, d_k, 2,dtype=torch.float32
        )
        inv_freq=theta**(-dimensions/d_k)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("sin_cache",angles.sin(),persistent=False)
        self.register_buffer("cos_cache",angles.cos(),persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
