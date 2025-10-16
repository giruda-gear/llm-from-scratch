import re

from ch02.simple_tokenizer_v1 import SimpleTokenizerV1


with open("ch02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

text = "Hello, world. Is this-- a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
# print(result)

# processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# processed = [item for item in processed if item.strip()]
processed = re.findall(r"\w+(?:'\w+)?|[.,:;!?()]", raw_text)
# print(len(processed))
# print(processed[:30])
all_words = sorted(set(processed))
vocab_size = len(all_words)
# print(vocab_size)
vocab = {token: integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
