import torch
import torch.nn as nn
from my_minGPT import GPT, GPTconfig


batch_size = 5
block_size = 1024
embedding_dim = 768
sequence = torch.randn(batch_size, block_size, embedding_dim)
print(sequence.shape)

