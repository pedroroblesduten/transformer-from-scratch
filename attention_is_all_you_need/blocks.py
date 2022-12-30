import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from helper import FeedForward, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, h_size, embedding_dim, n_heads=8, verbose=False):
        super(TransformerEncoderLayer, self).__init__()
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

        h = self.attention(x, x, x)
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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, h_size, embedding_dim, n_heads=8, mask=None, verbose=False):
        super(TransformerDecoderLayer, self).__init__()

        self.verbose = verbose
        self.mask = mask
        self.h_size = h_size
        self.inter_size = h_size//2
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.layer_norm1 = LayerNorm(h_size)
        self.layer_norm2 = LayerNorm(h_size)
        self.layer_norm3 = LayerNorm(h_size)
        self.masked_attention = MultiHeadAttention(self.embedding_dim, self.n_heads, self.mask)
        self.attention = MultiHeadAttention(self.embedding_dim, self.n_heads)
        self.feed_forward = FeedForward(self.h_size, self.inter_size)

    def forward(self, enc_out, dec_in):
        if self.verbose:
            print(f'Input shape for decoder layer: {dec_in.shape}')

        h = self.masked_attention(dec_in, dec_in, dec_in)
        if self.verbose:
            print(f'Shape after masked_attetion: {x.shape}')
            
        x = dec_in + h
        if self.verbose:
            print(f'Shape after sum of residual: {x.shape}')

        x = self.layer_norm1(x)
        if self.verbose:
            print(f'Shape after layer_norm1: {x.shape}')

        h = self.attention(enc_out, enc_out, x)
        if self.verbose:
            print(f'Shape after attention: {x.shape}')

        x = x + h
        if self.verbose:
            print(f'Shape after sum of residual: {x.shape}')

        x = self.layer_norm2(x)
        if self.verbose:
            print(f'Shape after layer_norm2: {x.shape}')

        h = self.feed_forward(x)
        if self.verbose:
            print(f'Shape after feed_forward: {x.shape}')

        x = x + h
        if self.verbose:
            print(f'Shape after sum of residual: {x.shape}')

        x = self.layer_norm3(x)
        if self.verbose:
            print(f'Shape after layer_norm3: {x.shape}')

        return x
