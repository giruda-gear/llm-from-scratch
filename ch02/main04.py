import torch

from ch02 import dataloader


# 2.7 creating token embedding
# vocab_size = 6
# output_dim = 3

# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)
# print(embedding_layer(torch.tensor[3]))

# 2.8 encoding word positions
with open("ch02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256
token_embediing_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

dataloader = dataloader.create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=4,
    shuffle=False,
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs)
print("-------")
print(targets)
print("\n inputs shape:", inputs.shape)

token_embeddings = token_embediing_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
