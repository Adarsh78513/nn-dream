import os

from tokenizer import Tokenizer

with open("tokenizerBPE/train.txt", "r", encoding='utf-8') as file:
    text = file.read()

vocab_size = 10000
token = Tokenizer()
token.train(text, vocab_size, verbose=True)

# save the model for later use
token.save("internetchunk")

# Breaking the string into tokens, to check what the tokenizer has learned
checkString = "This string will be broken into parts and all the indivisual tokens should not be on a character level"
print([token.decode([tokens]) for tokens in token.encode(checkString)])
print("Encoder & decoder working properly?", token.decode(token.encode(checkString)) == checkString)
# After using the vocab size 10000 (took 100 seconds to run), the checkString is tokenized like ['This ', 'string ', 'will be ', 'b', 'ro', 'k', 'en ', 'into ', 'par', 'ts ', 'and ', 'all the ', 'indi', 'visu', 'al ', 'to', 'k', 'en', 's ', 'should ', 'not be ', 'on a ', 'character ', 'le', 'v', 'el']
# This is good, it is learning, there is an issue, sometimes multiple words are getting clubbed together, like ...'will be '... or .....'not be'.... above, we dont want this, to solve this, use regex to break the text into groups and then train the transformer on this.
# Better tokenizer in tokenizerRegex.py