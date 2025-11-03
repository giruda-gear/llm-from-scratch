import torch

from ch03.data import inputs

# 3.3.2
# atten scores -> attention weights -> context vectors

# attn_scores = torch.empty(6,6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

attn_scores = inputs @ inputs.T  # matrix multiplication
print("attn_scores:\n", attn_scores)
# dim=-1 the last dimension of the tensor [rows, *columns*]
attn_weights = torch.softmax(attn_scores, dim=-1)
print("attn_weights:\n", attn_weights)

# row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# print("ROW 2 sum:", row_2_sum)
# print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
