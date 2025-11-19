import os
import re
from bisect import bisect_right
from itertools import accumulate
import torch
from torch.utils.data import Dataset
from nltk.tokenize import RegexpTokenizer

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

# TODO: This, and build_and_encode, shoudl be part fo tokenizer class
'''def tokenize(text, method='char'):
    assert method in ('char', 'word'), (
    f"Tokenization method must be 'char' or 'word' - got {method}")

    if method == 'word':
        tokenizer = RegexpTokenizer(r"\s+|<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
        tokens = tokenizer.tokenize(text)
        return tokens
    else: # method == 'char'
        return list(text)'''

def build_and_encode(lines, method='char'):
    print(f"Building vocabulary using {method} tokenization...")
    vocab = sorted(set(tokenize("".join(lines), method)))

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
        tokens = tokenize(line, method)
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

# TODO: test
class Tokenizer:
    def __init__(self, method='char'):

        self.oov_token_idx, self.oov_token     = 0, '<?>'
        self.start_token_idx, self.start_token = 1, '<s>'
        self.end_token_idx, self.end_token     = 2, '</s>'
        self.pad_token_idx, self.pad_token     = 3, '<>'

        self.method = None
        self._tokenize = None
        self.set_tokenizer(method)

        self.vocab_size = None
        self.idx2token = None
        self.token2idx = None
        #self.build_vocab(...)

    def set_tokenizer(self, method):
        assert method in ('char', 'word'), (
        f"Tokenization method must be 'char' or 'word' - got {method}")
        self.method = method
        if method == 'word':
            re_tokenizer = RegexpTokenizer(r"\s+|<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
            self._tokenize = lambda text: re_tokenizer.tokenize(text)
        else: # method == 'char'
            self._tokenize = lambda text: list(text)

    def build_vocab(self, sentences):
        assert isinstance(sentences, str) or (isinstance(sentences, list) and all(isinstance(sentence, str) for sentence in sentences)),\
            "Input must be a string or a list of strings"

        print(f"Building vocabulary using {self.method} tokenization...")
        if isinstance(sentences, str): sentences = [sentences]
        vocab = set()
        for sentence in sentences: vocab.update(self.tokenize(sentence))
        vocab = sorted(vocab)

        # TODO: Handle edge case where these appear organically in the text. For now, assume they don't
        # Care: adding indices outside of this order will fuck them up
        vocab.insert(self.oov_token_idx, self.oov_token)
        vocab.insert(self.start_token_idx, self.start_token)
        vocab.insert(self.end_token_idx, self.end_token)
        vocab.insert(self.pad_token_idx, self.pad_token)

        self.vocab_size = len(vocab)
        self.idx2token = dict(enumerate(vocab))
        self.token2idx = {token:idx for idx, token in self.idx2token.items()}
        print(f"Final vocabulary size: {self.vocab_size}\n")

    def tokenize(self, text):
        assert isinstance(text, str)
        return self._tokenize(text)

    # TODO: Type flexibility; should be able to encode / decode back and forth fluidly
    def encode(self, text_str):
        assert isinstance(text_str, str)
        return [self.token2idx.get(token, self.oov_token_idx) for token in self.tokenize(text_str)]
    
    # TODO: Type flexibility
    def decode(self, encoded_tensor):
        assert isinstance(encoded_tensor, torch.Tensor)
        return [self.idx2token.get(token_idx.item(), self.oov_token) for token_idx in encoded_tensor]