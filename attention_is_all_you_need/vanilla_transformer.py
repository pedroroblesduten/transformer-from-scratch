import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, h_size, embedding_dim, n_heads, mask=None, verbose=False):
        super().__init()

        self.mask = mask 
        self.h_size = h_size
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads



