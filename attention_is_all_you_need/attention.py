import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.linear_Q = nn.Linear(embedding_dim, head_dim)
        self.linear_K = nn.Linear(embedding_dim, head_dim)
        self.linear_V = nn.Linear(embedding_dim, head_dim)
    
    def scaled_dot_product_attetion(self, query, key, value, mask, dropout=None):
        d_k = query.size(-1)

        # Calculating Q*V/sqrt(d_k)
        scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        output = torch.bmm(p_attn, value)

        return output

    def forward(self, Q, K, V, mask):
        one_head_attetion = self.scaled_dot_product_attetion(
            self.linear_Q(Q),
            self.linear_K(K),
            self.linear_V(V),
            mask)
        
        return one_head_attetion

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout=0.1, verbose=False):
        super().__init__()
        assert embedding_dim % n_heads == 0
        self.verbose = True
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads        
        self.heads = nn.ModuleList([
            AttentionHead(self.embedding_dim, self.head_dim)
            for _ in range(self.n_heads)
        ])
        self.output_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    def forward(self, Q, K, V, mask=None):
        x = torch.cat([head(Q, K, V, mask) for head in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
        

