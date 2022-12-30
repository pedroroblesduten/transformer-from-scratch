import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from helper import FeedForward, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, h_size, embedding_dim, n_heads, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.h_size = h_size
        self.inter_size  = h_size // 2
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.layer_norm1 = LayerNorm(h_size)
        self.layer_norm2 = LayerNorm(h_size)
        self.attention = MultiHeadAttention(self.embedding_dim, self.n_heads)
        self.feed_forward = FeedForward(self.h_size, self.inter_size)
        
    def forward(self, x):
        if self.verbose:
            print(f'Input shape for the encoder layer: {x.shape}')

        h = self.attention(x)
        if self.verbose:
            print(f'Shape after MultiHeadAttention: {h.shape}')

        x = x + h
        if self.verbose:
            print(f'Shape after sum of residual: {x.shape}')

        x = self.layer_norm1(x)
        if self.verbose:
            print(f'Shape after layer_norm1: {x.shape}')

        h = self.feed_forward(x)
        if self.verbose:
            print(f'Shape after feed_forward: {h.shape}')

        x = x + h
        if self.verbose:
            print(f'Shape after sum of residual: {x.shape}')

        x = self.layer_norm2(x)
        if self.verbose:
            print(f'Shape after layer_norm2: {x.shape}')

        return x


