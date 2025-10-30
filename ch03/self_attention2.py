import torch


inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],  # step (x^6)
    ]
)

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