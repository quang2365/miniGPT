import torch
import torch.nn as nn
import torch.nn.functional as F
def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple
class SwiGLU(nn.Module):
    def __init__(self,dim,hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8*dim/3)
            hidden_dim = round_up(hidden_dim,64)
        self.gate = nn.Linear(dim,hidden_dim,bias=False)

        self.up = nn.Linear(dim,hidden_dim,bias=False)

        self.down = nn.Linear(hidden_dim,dim,bias=False)

    def forward(self,x):
        gate = F.silu(self.gate(x))

        up = self.up(x)

        out = gate * up

        down = self.down(out)

        return down
