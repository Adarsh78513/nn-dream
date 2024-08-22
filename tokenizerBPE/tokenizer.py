import unicodedata

def byte_pair_counts(ids):
    byte_pair_counts={}
    """Count frequency of all consecutive symbol pairs in list ids, and return a dictionary of counts"""
    for i in range(1, len(ids)):
        pair = (ids[i-1], ids[i])
        if pair in byte_pair_counts:
            byte_pair_counts[pair] += 1
        else:
            byte_pair_counts[pair] = 1
    return byte_pair_counts

def merge(ids, byte_pair, new_id):
    """Merge all occurrences of the byte pair with new ids in the list ids"""
    new = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i+1]) == byte_pair:
            new.append(new_id)
            i += 2
        else:
            new.append(ids[i])
            i += 1
    return new

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer:
    """Base tokenizer class, this class should be inherited by other tokenizer classes, 
    using just like this it will just merge all byte pairs in the vocab with no special conditions"""
    def __init__(self):
        self.merges = {} # dict (p0, p1) -> c
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()
        
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        for i in range(vocab_size - 256):
            pairs_count = byte_pair_counts(tokens)
            # print(pairs_count)
            if not pairs_count:
                break
            pair = max(pairs_count, key=pairs_count.get)
            if verbose:
                print(f"Iteration {i}")
                print("merging", pair, "to", 256 + i)
            tokens = merge(tokens, pair, 256 + i)
            # print(tokens)
            self.merges[pair] = 256 + i
            self.vocab[256 + i] = self.vocab[pair[0]] + self.vocab[pair[1]]
        # self.vocab = self._build_vocab()

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            pairs_count = byte_pair_counts(tokens)
            pair = min(pairs_count, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        tokens = [self.vocab[i] for i in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")
    
    def _build_vocab(self):
        vocab = {i:bytes([i]) for i in range(256)} # Initialize vocab with all byte values
        for (parent0, parent1) , idx in self.merges.items():
            vocab[idx] = vocab[parent0] + vocab[parent1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special
        return vocab
    
    def save(self, file_name):
        """
        Saving the model and the vocabulary
        """
        save_location = "tokenizer_models/"
        model_file = save_location + file_name + ".model"
        vocab_fille = save_location + file_name + ".vocab"
        with open(model_file, 'w') as f:
            f.write("tokenizer.v1")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_fille, 'w', encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
                