import os

from tokenizer import Tokenizer

with open("tokenizerBPE/train.txt", "r", encoding='utf-8') as file:
    text = file.read()

vocab_size = 512
t = Tokenizer()
print("encoded", len(t.encode("hello there")))
t.train(text, vocab_size)
print("encoded", len(t.encode("hello there")))