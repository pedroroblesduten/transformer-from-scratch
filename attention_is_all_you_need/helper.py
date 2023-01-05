import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import AutoTokenizer, AutoConfig

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
    

class PositionalEnconding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(max_len, d_model)
        self.position = torch.arange(0, max_len, 1).unsqueeze(1)
        self.div_term  = torch.pow(10000, 2 * (torch.arange(0, d_model, 2)/d_model))
       
    def forward(self, x):
        self.pe[:, 0::2] = torch.sin(self.position*self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position*self.div_term)
        x = x + self.pe[: x.shape[1]].requires_grad_(False)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embeddings(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)*sqrt(self.d_model)

class TokenizerHuggingFace(nn.Module):
    def __init__(self, model_name='bert-base-uncased', special_tokens=True):
        super().__init__()
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.special_tokens = special_tokens

    def forward(self, text):

        text = self.tokenizer(text, return_tensors='pt', add_special_tokens=self.special_tokens)
        token_emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        text_emb = token_emb(text.input_ids)

        return text_emb

class Generator(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        x = F.log_softmax(x, dim=-1)
        return x
