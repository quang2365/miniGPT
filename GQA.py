import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from RoPE import RoPE
from LoRALinear import LoRALinear
def RepeatKV(x,n_rep):
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x,repeats=n_rep,dim=1)

class GroupedQueryAttention(nn.Module):
    def __init__(self,c_emb,n_query_heads,n_key_value_heads, max_seq_len):
        super().__init__()
        assert c_emb % n_query_heads == 0, "c_emb phải chia hết cho n_query_heads"
        assert n_query_heads % n_key_value_heads == 0, "n_query_heads phải chia hết cho n_key_value_heads"
        self.dim = c_emb

        self.n_query_heads = n_query_heads

        self.n_key_value_heads = n_key_value_heads

        self.head_dim = c_emb//n_query_heads

        self.group_size = n_query_heads//n_key_value_heads

        self.q_projec = nn.Linear(self.dim, self.head_dim * self.n_query_heads,bias=False)

        self.k_projec = nn.Linear(self.dim, self.head_dim * self.n_key_value_heads,bias=False)

        self.v_projec = nn.Linear(self.dim, self.head_dim * self.n_key_value_heads,bias=False)

        self.rope = RoPE(self.head_dim,max_seq_len)

        self.f_out_projec = nn.Linear(self.dim,self.dim,bias=False)

    def forward(self,x):

        B,T,C = x.shape

        q = self.q_projec(x)

        k = self.k_projec(x)

        v = self.v_projec(x)

        q = q.view(B,T,self.n_query_heads,self.head_dim)

        k = k.view(B,T,self.n_key_value_heads,self.head_dim)

        v = v.view(B,T,self.n_key_value_heads,self.head_dim)

        q = q.transpose(2,1)
        k = k.transpose(2,1)
        v = v.transpose(2,1)

        q = self.rope(q)

        k = self.rope(k)

        k = RepeatKV(k,self.group_size)

        v = RepeatKV(v,self.group_size)

        scorse = q @ k.transpose(-2,-1)

        scorse = scorse/math.sqrt(self.head_dim)

        casual_mask = torch.tril(torch.ones(T,T,dtype = torch.bool,device=x.device))

        scorse = scorse.masked_fill(~casual_mask,float('-inf'))

        attention_weight = F.softmax(scorse,dim=-1)

        out = (attention_weight @ v)

        out = out.transpose(1,2)

        out = out.contiguous().view(B,T,C)

        out = self.f_out_projec(out)

        return out


        
