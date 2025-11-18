import os
import re
from bisect import bisect_right
from itertools import accumulate
import torch
from torch.utils.data import Dataset
from nltk.tokenize import RegexpTokenizer

#import nltk
#from nltk import sent_tokenize
#from nltk.corpus import brown
#nltk.data.path.append(os.path.dirname(os.path.dirname(__file__)))
base_path = os.path.dirname(os.path.abspath(__file__))

def preprocess_akjv(include_book=True):
    source_filepath = os.path.join('..', 'datasets', 'akjv.txt')
    # utf-8-sig strips leading BOM char
    with open(source_filepath, encoding="utf-8-sig") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]


    pattern = re.compile(r'^\s*(?P<book>.+?)\s+(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.+?)\s*$')
    processed_lines = []
    for line in lines:
        match = pattern.match(line)
        if not match:
            print(f"Line '{line}' discarded")
            continue
        if include_book:
            processed = f"{match['book']}\t{match['text']}"
        else:
            processed = f"{match['text']}"
        processed_lines.append(processed)
    return processed_lines

def tokenize(text, tokenization='char'):
    assert tokenization in ('char', 'word'), (
    f"tokenization must be 'char' or 'word' - got {tokenization}")

    if tokenization == 'word':
        tokenizer = RegexpTokenizer(r"\s+|<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
        tokens = tokenizer.tokenize(text)
        return tokens
    else: # tokenization == 'char'
        return list(text)

def build_and_encode(lines, tokenization='char'):
    print(f"Building vocabulary using {tokenization} tokenization...")
    vocab = sorted(set(tokenize("".join(lines), tokenization)))

    vocab.insert(0, '<?>') # out of dictionary token
    vocab.insert(1, '<s>') # start token
    vocab.insert(2, '</s>') # end token
    vocab.insert(3, '<>') # pad token

    vocab_size = len(vocab)
    idx2token = dict(enumerate(vocab))
    token2idx = {word:i for i, word in idx2token.items()}
    print(f"Final vocabulary size: {vocab_size}\n")

    print("Encoding Lines...")
    encoded_lines = []
    for line in lines:
        tokens = tokenize(line, tokenization)
        encoded = [token2idx.get(token, 0) for token in tokens]
        encoded_lines.append([1] + encoded + [2])
    print(f"Finished encoding {len(encoded_lines)} lines\n")
    return encoded_lines, vocab_size, idx2token, token2idx

class TransformerDataset(Dataset):
    def __init__(self, encoded_lines, context_len,
                 start_token_idx=1, end_token_idx=2, pad_token_idx=3):
        assert len(encoded_lines) >= 1 and context_len >= 1
        assert all(
            len(line) > 2 and
            line[0] == start_token_idx and
            line[-1] == end_token_idx
            for line in encoded_lines
        )

        self.lines = encoded_lines
        self.context_len = context_len
        self.pad_token_idx = pad_token_idx

        counts = [len(line)-1 for line in self.lines]
        self.cumulative_counts = list(accumulate(counts))

    def __getitem__(self, idx):
        assert (idx >=0 and idx < self.cumulative_counts[-1])

        line_idx = bisect_right(self.cumulative_counts, idx)
        p_0 = self.cumulative_counts[line_idx-1] if line_idx > 0 else 0

        x_f = idx - p_0 + 1 # +1 because slices don't include last index
        y_f = x_f + 1

        x_i = max(x_f - self.context_len, 0)
        y_i = max(y_f - self.context_len, 0)

        pad_x = max(self.context_len - x_f,0)
        pad_y = max(self.context_len - y_f,0)

        x = [self.pad_token_idx]*pad_x + self.lines[line_idx][x_i:x_f]
        y = [self.pad_token_idx]*pad_y + self.lines[line_idx][y_i:y_f]

        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.cumulative_counts[-1]

