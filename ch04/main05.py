import tiktoken
import torch

from ch04.gpt_config import GPT_CONFIG_124M
from ch04.gpt_model import GPTModel, generate_text_simple


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

print("input batch:\n", batch)
print("\noutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"total number of parameters: {total_params}")

print("token embedding layer shape:", model.tok_emb.weight.shape)
print("output layer shape:", model.out_head.weight.shape)

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"number of trainable parameters considering weight tying: {total_params_gpt2}")

total_size_bytes = total_params * 4  # float32(4 byte)
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"total size of the model: {total_size_mb}")


start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
model.eval()
output = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("output:", output)
print("output length:", len(output[0]))

decoded_text = tokenizer.decode(output.squeeze(0).tolist())
print(decoded_text)