import torch 
import torch.nn as nn
import math
class RMSNorm(nn.Module):
    def __init__(self,dim, esp = 1e-6):
        super().__init__()
        self.esp = esp
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self,x):

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.esp)

        return x_norm * self.weight
