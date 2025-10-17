import tiktoken


with open("ch02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2.5 Byte pair encoding
tiktokenizer = tiktoken.get_encoding("gpt2")
text2 = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)
# text2 = "Akwirw ier"
# integers = tiktokenizer.encode(text2, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tiktokenizer.decode(integers)
# print(strings)

# 2.6 Data sampling with a sliding window
enc_text = tiktokenizer.encode(raw_text)
print(len(enc_text))  # 5145

enc_sample = enc_text[50:]

context_size = 4
# input-target pairs that we can turn into use for the LLM training
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"x: {x}")  # x: [290, 4920, 2241, 287]
print(f"y:      {y}")  # y:      [4920, 2241, 287, 257]


for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "--->", desired)  # [290, 4920, 2241, 287] ---> 257

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tiktokenizer.decode(context), "--->", tiktokenizer.decode([desired]))
