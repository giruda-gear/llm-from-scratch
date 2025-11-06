# 3.6.2 Implementing multi-head attention with weight splits
import torch
from ch03.data import inputs
from ch03.multi_head_attention import MultiHeadAttention


torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha2 = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs2 = mha2(batch)
print(context_vecs2)
print("context_vecs2.shape:", context_vecs2.shape)

a = torch.tensor(
    [
        [
            [
                [0.2745, 0.6584, 0.2775, 0.8573],
                [0.8993, 0.0390, 0.9268, 0.7388],
                [0.7179, 0.7058, 0.9156, 0.4340],
            ],
            [
                [0.0772, 0.3565, 0.1479, 0.5331],
                [0.4066, 0.2318, 0.4545, 0.9737],
                [0.4606, 0.5159, 0.4220, 0.5786],
            ],
        ]
    ]
)
print(a.shape)  # torch.Size([1, 2, 3, 4]) batch, head, context_length, dim
print(a @ a.transpose(2, 3)) # attn_scores = queries @ keys.transpose(2, 3)

first_head = a[0, 0, :, :]
# ([[0.2745, 0.6584, 0.2775, 0.8573],       ([[0.2745, 0.8993, 0.7179],
# [0.8993, 0.0390, 0.9268, 0.7388],     @    [0.6584, 0.0390, 0.7058],
# [0.7179, 0.7058, 0.9156, 0.4340]])         [0.2775, 0.9268, 0.9156],
#                                            [0.8573, 0.7388, 0.4340]])
first_res = first_head @ first_head.T
print("first head:\n", first_res)

second_head = a[0, 1, :, :]
print(second_head)
second_res = second_head @ second_head.T
print("\nsecond head:\n", second_res)
