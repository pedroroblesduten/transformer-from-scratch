import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import EncoderDecoder
from helper import PositionalEnconding, Generator

class Transformer(nn.Module):
    def __init__(self, h_size, embedding_dim, n_heads=8, mask=None, verbose=False, N=6):
        super().__init()        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.encoder_decoder = EncoderDecoder(embedding_dim, n_heads, verbose=verbose, N=N)
        self.post_enc = PositionalEnconding(embedding_dim, dropout=0.01)
        
    def forward(self, src, trg, trg_mask):
        src = post_enc(x)
        trg = post_enc(x)


