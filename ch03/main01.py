import torch

from ch03.data import inputs


# 3.3.1 A simple self-attention mechanism without trainable weights
query = inputs[1]  # journey
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("attention score:", attn_scores_2)
# [(your <- journey), (journey <- journey), (starts <- journey), ...]

# understanding dot product
# your <- journey
# [0.43, 0.15, 0.89] * [0.55
#                       0.87
#                       0.66]
res = 0
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
# print(res) # 0.9544

# normalized attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print(attn_weights_2_tmp)
# print("sum:", attn_weights_2_tmp.sum())  # 1.0000


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("attention weights:", attn_weights_2_naive)
# print("sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("attention weights(softmax):", attn_weights_2)
# print("sum:", attn_weights_2.sum())

print(attn_weights_2)
context_vec_2 = torch.zeros(query.shape)  # tensor([0., 0., 0.])
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
    print(attn_weights_2[i], "*", x_i)
print("context vector:", context_vec_2)
