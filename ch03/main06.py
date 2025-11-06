import torch

from ch03.data import inputs
from ch03.multi_head_attention_wrapper import MultiHeadAttentionWrapper

# 3.6.1
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
