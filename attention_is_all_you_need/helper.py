import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class FeedForward(nn.Module):
    def __init__(self, h_size, inter_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(h_size, inter_size)
        self.linear2 = nn.Linear(inter_size, h_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = self.a_2*(x-mean)/(std+self.eps) + self.b_2
        return output





