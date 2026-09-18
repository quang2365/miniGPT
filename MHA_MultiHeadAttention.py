import torch
import torch.nn as nn
import torch.nn.functional as F
class Single_head_attention(nn.Module):
    def __init__(self,n_emb_C, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_emb_C,head_size,bias = False)
        self.key = nn.Linear(n_emb_C,head_size,bias = False)
        self.value = nn.Linear(n_emb_C,head_size,bias = False)

        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    def forward(self,x):
        B,T,H = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # can use T insteal self.head_size
        attention_weight = (q @ (k.transpose(-2,-1)))/self.head_size ** 0.5

        attention_weight = attention_weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        attention_weight = F.softmax(attention_weight, dim = -1)

        output = attention_weight @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,n_emb_C,n_head_H,block_size):
        super().__init__()
        assert( n_emb_C % n_head_H == 0 )
        head_size = n_emb_C // n_head_H

        self.heads = nn.ModuleList([Single_head_attention(n_emb_C,head_size,block_size) for _ in range(n_head_H)])
        #use this after concat
        self.projection = nn.Linear(n_emb_C,n_emb_C)

    def forward(self,x):

        out = torch.cat([head(x) for head in self.heads], dim= -1
        
    ) 

        return self.projection(out)
