import torch
import torch.nn as nn
class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len,base = 10000):
        super().__init__()
        inv_freq = 1.0/(base ** (torch.arange(0,head_dim,2).float()/head_dim))

        position = torch.arange(max_seq_len).float()

        freq = torch.outer(position,inv_freq)

        self.register_buffer('cos_cache',torch.cos(freq),persistent=True)

        self.register_buffer('sin_cache',torch.sin(freq),persistent=True)

    def forward(self,x):
        B,heads,T,H = x.shape

        cos = self.cos_cache[:T]

        sin = self.sin_cache[:T]

        cos = cos [None,None,:,:]

        sin = sin [None,None,:,:]

        x_even = x[...,0::2]

        x_odd = x[...,1::2]

        rotated_even = x_even * cos - x_odd * sin

        rotated_odd = x_even * sin + x_odd * cos

        x_rotated = torch.stack((rotated_even,rotated_odd),dim = -1)

        x_rotated = x_rotated.flatten(-2)

        return x_rotated