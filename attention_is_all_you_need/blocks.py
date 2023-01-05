import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from helper import FeedForward, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads=8, verbose=False):
        super(TransformerEncoderLayer, self).__init__()
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.inter_size  = embedding_dim // 2
        self.n_heads = n_heads

        self.layer_norm1 = LayerNorm(embedding_dim)
        self.layer_norm2 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(self.embedding_dim, self.n_heads)
        self.feed_forward = FeedForward(self.embedding_dim, self.inter_size)
        
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
    def __init__(self, embedding_dim, n_heads=8, verbose=False):
        super(TransformerDecoderLayer, self).__init__()

        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.inter_size = embedding_dim//2
        self.n_heads = n_heads
        
        self.layer_norm1 = LayerNorm(embedding_dim)
        self.layer_norm2 = LayerNorm(embedding_dim)
        self.layer_norm3 = LayerNorm(embedding_dim)
        self.masked_attention = MultiHeadAttention(self.embedding_dim, self.n_heads)
        self.attention = MultiHeadAttention(self.embedding_dim, self.n_heads)
        self.feed_forward = FeedForward(self.embedding_dim, self.inter_size)

    def forward(self, enc_out, dec_in, mask):
        if self.verbose:
            print(f'Input shape for decoder layer: {dec_in.shape}')

        h = self.masked_attention(dec_in, dec_in, dec_in, mask)
        if self.verbose:
            print(f'Shape after masked_attetion: {h.shape}')
            
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


class EncoderDecoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, verbose, N):
        super().__init__()
        self.verbose = verbose
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embedding_dim, verbose=self.verbose) for _ in range(N)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(embedding_dim, verbose=self.verbose) for _ in range(N)])
    
    def _make_mask(self, y):
        batch, y_len, emb_dim = y.shape
        mask = torch.tril(torch.ones(y_len, y_len)).unsqueeze(0)
        return mask.reshape(batch, y_len, y_len)

    def forward(self, x, y):
        for enc in self.encoder_layers:
            x = enc(x)
        print(f'Shape de saida do encoder: {x.shape}')
    
        for dec in self.decoder_layers:
            y = dec(x, y, self._make_mask(y))
        print(f'Shape de saída do decoder: {y.shape}')

        return y



    

