import torch

from ch03.causal_attention import CausalAttention
from ch03.data import inputs


# 3.5.3 Implementing a compact causal attention class
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
# print("batch:", batch)
d_in = inputs.shape[1]
d_out = 2
context_length = batch.shape[1]

ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
