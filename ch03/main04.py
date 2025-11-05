import torch

from ch03.data import inputs
from ch03.self_attention_v2 import SelfAttention_v2


# 3.5 Hiding future words with casual attention
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)
print("attn_weights:\n", attn_weights)
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("mask_simple:\n", mask_simple)
# the elements above the diagonal are zeroed out
masked_simple = attn_weights * mask_simple
print("masked_simple:\n", masked_simple)

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("masked_simple_norm:\n", masked_simple_norm)

# masking trick e^{-inf} = 0
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked:", masked)

attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print("atten_weights:\n", attn_weights)

# torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
print("dropout(attn_weights):\n", dropout(attn_weights))
