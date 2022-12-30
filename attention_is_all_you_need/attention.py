import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import scaled_dot_product_attetion

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.linear_Q = nn.Linear(embedding_dim, head_dim)
        self.linear_K = nn.Linear(embedding_dim, head_dim)
        self.linear_V = nn.Linear(embedding_dim, head_dim)

    def forward(self, input):
        one_head_attetion = scaled_dot_product_attetion(
            self.linear_Q(input),
            self.linear_K(input),
            self.linear_V(input))
        
        return one_head_attetion

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout=0.1, verbose=False):
        super().__init__()
        self.verbose = True
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads        
        self.heads = nn.ModuleList([
            AttentionHead(self.embedding_dim, self.head_dim)
            for _ in range(self.n_heads)
        ])
        self.output_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    def forward(self, h):
        x = torch.cat([head(h) for head in self.heads], dim = -1)
        x = self.output_linear(x)
        return x
        

