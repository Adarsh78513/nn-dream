import os

from tokenizer import Tokenizer

with open("tokenizerBPE/train.txt", "r", encoding='utf-8') as file:
    text = file.read()

vocab_size = 512
t = Tokenizer()
print("encoded", len(t.encode("hello there")))
t.train(text, vocab_size)
print("encoded length", len(t.encode("hello thein")))
checkString = "Hello there"
print("Encoder decoder working properly:", t.decode(t.encode(checkString)) == checkString)