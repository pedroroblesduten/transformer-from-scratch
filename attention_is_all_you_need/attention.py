import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attetion(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # Calculating Q*V/sqrt(d_k)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, 1e-9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.bmm(p_attn, value)

    return output

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim, mask=None):
        super().__init__()
        self.mask = mask
        self.linear_Q = nn.Linear(embedding_dim, head_dim)
        self.linear_K = nn.Linear(embedding_dim, head_dim)
        self.linear_V = nn.Linear(embedding_dim, head_dim)

    def forward(self, Q, K, V):
        one_head_attetion = scaled_dot_product_attetion(
            self.linear_Q(Q),
            self.linear_K(K),
            self.linear_V(V),
            mask=self.mask)
        
        return one_head_attetion

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout=0.1, mask=None, verbose=False):
        super().__init__()
        self.mask = mask
        self.verbose = True
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads        
        self.heads = nn.ModuleList([
            AttentionHead(self.embedding_dim, self.head_dim, self.mask)
            for _ in range(self.n_heads)
        ])
        self.output_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    def forward(self, Q, K, V):
        x = torch.cat([head(Q, K, V) for head in self.heads], dim = -1)
        x = self.output_linear(x)
        return x
        

