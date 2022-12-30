from transformers import AutoTokenizer, AutoConfig
import torch 
import torch.nn as nn
from blocks import TransformerEncoderLayer

model_ckpt = 'bert-base-uncased'
tokeninzer = AutoTokenizer.from_pretrained(model_ckpt)
text = "time flies like an arrow"
inputs = tokeninzer(text, return_tensors='pt', add_special_tokens=False)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
inputs_emb = token_emb(inputs.input_ids)
h_size = config.hidden_size
embedding_dim = config.hidden_size
n_heads = 8

encoder_layer = TransformerEncoderLayer(h_size, embedding_dim, n_heads)
size = encoder_layer(inputs_emb).size()
print(encoder_layer)
print(size)
print('--')



print(text)
print(inputs.input_ids)
print(token_emb)
print(inputs_emb.size())
