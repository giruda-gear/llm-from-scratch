import torch

from ch04.feed_forward import FeedForward
from ch04.gpt_config import GPT_CONFIG_124M


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)