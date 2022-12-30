import torch
import torch.nn as nn
import torch.nn.functionall as F


def scaled_dot_product_attetion(query, key, value, mask=None, dropout=None)
    d_k = query.size(-1)
    # Calculating Q*V/sqrt(d_k)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, 1e-9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.bmm(p_attn, value)

    return output, p_attn
