from transformers import AutoTokenizer, AutoConfig
import torch 
import torch.nn as nn
from blocks import TransformerEncoderLayer, TransformerDecoderLayer

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
enc_out = encoder_layer(inputs_emb)
print(f'Shape de saida do encoder: {enc_out.shape}')

decoder_layer = TransformerDecoderLayer(h_size, embedding_dim, n_heads)
dec_out = decoder_layer(enc_out, inputs_emb)
print(f'Shape de saida do decoder: {dec_out.shape}')

print('--- ENCODER LAYER --')
print(encoder_layer)
print('-- DECODER LAYER --')
print(decoder_layer)

