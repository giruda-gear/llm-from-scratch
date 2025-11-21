import torch
import tiktoken

from ch04.gpt_model import GPTModel, generate_text_simple
from ch05.gpt_config import GPT_CONFIG_124M
from ch05.utility import text_to_token_ids, token_ids_to_text


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("output text:\n", token_ids_to_text(token_ids, tokenizer))

# "every effort moves"
# "I really like"
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
# token IDs we want the LLM to generate
# "    effort moves you"
# "   really like chocolate"
targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)  # disable gradient tracking since we are not training yet
probas = torch.softmax(logits, dim=-1)  # probability of each token in vocabulary
print(probas.shape)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("token IDs:\n", token_ids)
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


text_idx = 0  # first batch
# probas[0, 0, 3626] = 7.4541e-05  (every -> effort)
# probas[0, 1, 6100] = 3.1061e-05  (effort -> moves)
# probas[0, 2, 345] = 1.1563e-05   (moves -> you)
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("log_probas:", log_probas)
avg_log_probas = torch.mean(log_probas)
print("avg_log_probas:", avg_log_probas)
neg_avg_log_probas = -1 * avg_log_probas
print("neg_avg_log_probas:", neg_avg_log_probas)

print("logits.shape:", logits.shape)  # batch size, number of tokens, vocabulary size
print("targets.shape:", targets.shape)  # batch size, number of tokens

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("flattened logits:", logits_flat.shape)
print("flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("loss:", loss)
