import re


from ch02.simple_tokenizer_v1 import SimpleTokenizerV1
from ch02.simple_tokenizer_v2 import SimpleTokenizerV2


with open("ch02/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# text = "Hello, world. Is this-- a test."
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# result = [item for item in result if item.strip()]
# print(result)


processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
processed = [item for item in processed if item.strip()]
# processed = re.findall(r"\w+(?:'\w+)?|[.,:;!?()]", raw_text)
# print(len(processed))
# print(processed[:30])

all_words = sorted(set(processed))
vocab_size = len(all_words)
print(vocab_size)
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(list(vocab.items())[:10]):
    print(item)


tokenizer_v1 = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = tokenizer_v1.encode(text)
print(ids)
print(tokenizer_v1.decode(ids))

all_tokens = all_words
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
all_vocab = {t: i for i, t in enumerate(all_tokens)}
print(len(all_vocab))

for i, item in enumerate(list(all_vocab.items())[-5:]):
    print(item)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
tokenizer_v2 = SimpleTokenizerV2(all_vocab)
print(tokenizer_v2.encode(text))
print(tokenizer_v2.decode(tokenizer_v2.encode(text)))
