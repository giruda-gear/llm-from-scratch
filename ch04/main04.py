import torch

from ch04.transformer_block import TransformerBlock
from ch04.gpt_config import GPT_CONFIG_124M


torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

# the transformer architecture processes sequences of data 
# without altering their shape throughout the network.
print("input shape:", x.shape)
print("output shape:", output.shape)