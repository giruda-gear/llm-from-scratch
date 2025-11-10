import torch
import tiktoken

from ch04.dummy_gpt_model import DummyGPTModel
from ch04.gpt_config import GPT_CONFIG_124M
from ch04.layer_norm import LayerNorm

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("output shape:", logits.shape)
print(logits)

# 4.2 Normalizing activations with layer normalization
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
nn = torch.nn
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("mean:\n", mean)
print("variation:\n", var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)
print("normalized layer outputs:\n", out_norm)
print("mean:\n", mean)
print("variance:\n", var)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean_ln = out_ln.mean(dim=-1, keepdim=True)
var_ln = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("mean_ln:\n", mean_ln)
print("var_ln:\n", var_ln)
