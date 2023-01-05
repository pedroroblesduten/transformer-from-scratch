from transformers import AutoTokenizer, AutoConfig
import torch 
import torch.nn as nn
from blocks import TransformerEncoderLayer, TransformerDecoderLayer, EncoderDecoder
from helper import TokenizerHuggingFace, PositionalEnconding

model_ckpt = 'bert-base-uncased'
tokeninzer = AutoTokenizer.from_pretrained(model_ckpt)
text = "time flies like an arrow"
inputs = tokeninzer(text, return_tensors='pt', add_special_tokens=False)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
inputs_emb = token_emb(inputs.input_ids)
embedding_dim = config.hidden_size
n_heads = 8

print(inputs_emb)
my_tokenizer = TokenizerHuggingFace(special_tokens=False)
text_emb = my_tokenizer(text)
print(text_emb)

print(f'Shape tokenizer livro: {inputs_emb.shape}')
print(f'Shape my tokenizer: {text_emb.shape}')

encoder_decoder = EncoderDecoder(embedding_dim, n_heads, verbose=True)
out = encoder_decoder(inputs_emb, inputs_emb)
print('Shape de saida do encoder_decoder')


post_enc = PositionalEnconding(embedding_dim, 0.01)
token_pe = post_enc(text_emb)
print(token_pe)
print(token_pe.shape)

print('--- ENCODER LAYER --')
# print(encoder_layer)
print('-- DECODER LAYER --')
# print(decoder_lay
