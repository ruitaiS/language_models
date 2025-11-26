import os
import re
import random
from bisect import bisect_right
from itertools import accumulate
import torch
from torch.utils.data import Dataset
from nltk.tokenize import RegexpTokenizer

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

def build_train_val_loaders(dataset, batch_size, validation_p, verbose=True, **kwargs):
    # shuffle, drop_last, num_workers, pin_memory, prefetch_factor, persistent_workers
    val_size = int(len(dataset) * validation_p)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=kwargs.pop("shuffle", True),
        drop_last=kwargs.pop("drop_last", True),
        num_workers=kwargs.pop("num_workers", 4), # or 8
        pin_memory=kwargs.pop("pin_memory", True),
        prefetch_factor=kwargs.pop("prefetch_factor", 2), # to 4
        persistent_workers=kwargs.pop("persistent_workers", True))

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=kwargs.pop("shuffle", True),
        drop_last=kwargs.pop("drop_last", True),
        num_workers=kwargs.pop("num_workers", 4), # or 8
        pin_memory=kwargs.pop("pin_memory", True),
        prefetch_factor=kwargs.pop("prefetch_factor", 2), # to 4
        persistent_workers=kwargs.pop("persistent_workers", True))

    if verbose:
        print(f"Dataset Total Sequences: {len(dataset)} || Validation Proportion: {validation_p}")
        print(f"Training Loader Sequences: {len(train_loader) * batch_size}")
        print(f"Validation Loader Sequences: {len(val_loader) * batch_size}")
        print(f"Sum: {len(train_loader) * batch_size + len(val_loader) * batch_size}\n")
        # print non-standard kwargs if you want

    return train_loader, val_loader

class Tokenizer:
    def __init__(self, method='char', init_text=None):

        self.oov_token   = '<?>'
        self.start_token = '<s>'
        self.end_token   = '</s>'
        self.pad_token   = '<>'

        self.method    = None
        self._tokenize = None
        self.set_tokenizer(method)

        self.oov_token_idx   = None
        self.start_token_idx = None
        self.end_token_idx   = None
        self.pad_token_idx   = None
        self.vocab_size      = None
        self.idx2token       = None
        self.token2idx       = None
        if init_text:self.build_vocab(init_text)

    def cfg(self):
        return {
                'vocab_size': self.vocab_size,
                'oov_token_idx': self.oov_token_idx,
                'start_token_idx': self.start_token_idx,
                'end_token_idx': self.end_token_idx,
                'pad_token_idx': self.pad_token_idx,
                'oov_token': self.oov_token,
                'start_token': self.start_token,
                'end_token': self.end_token,
                'pad_token': self.pad_token,
                }

    def bpe(text_str):
        # TODO
        pass

    def set_tokenizer(self, method):
        assert method in ('char', 'word'), (
        f"Tokenization method must be 'char' or 'word' - got {method}")
        self.method = method
        if method == 'word':
            re_tokenizer = RegexpTokenizer(r"\s+|<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
            self._tokenize = lambda text: re_tokenizer.tokenize(text)
        elif method == 'char':
            self._tokenize = lambda text: list(text)
        else: # bpe
            self._tokenize = lambda text: self.bpe(text)

    def build_vocab(self, text):
        assert isinstance(text, str) or (isinstance(text, list) and all(isinstance(line, str) for line in text)),\
            "Input must be a string or a list of strings"
        if isinstance(text, str): text = [text]

        print(f"Building vocabulary using {self.method} tokenization...")
        vocab = set()
        for line in text: vocab.update(self.tokenize(line))
        vocab = sorted(vocab)

        # TODO: Handle edge case where these appear organically in the text. For now, assume they don't
        vocab.insert(0, self.oov_token)
        vocab.insert(1, self.start_token)
        vocab.insert(2, self.end_token)
        vocab.insert(3, self.pad_token)

        self.oov_token_idx   = 0
        self.start_token_idx = 1
        self.end_token_idx   = 2
        self.pad_token_idx   = 3

        self.vocab_size = len(vocab)
        self.idx2token  = dict(enumerate(vocab))
        self.token2idx  = {token:idx for idx, token in self.idx2token.items()}
        print(f"Final vocabulary size: {self.vocab_size}\n")

    def tokenize(self, text_str):
        assert isinstance(text_str, str)
        return self._tokenize(text_str)

    def encode(self, text_str):
        assert isinstance(text_str, str)
        return [self.token2idx.get(token, self.oov_token_idx) for token in self.tokenize(text_str)]
    
    def encode_lines(self, lines):
        assert isinstance(lines, str) or (isinstance(lines, list) and all(isinstance(line, str) for line in lines)),\
            "Input must be a string or a list of strings"
        if isinstance(lines, str): lines = [lines]

        encoded_lines = []
        for line in lines:
            idx_seq = [self.start_token_idx] + self.encode(line) + [self.end_token_idx]
            encoded_lines.append(idx_seq)
        print(f"Finished encoding {len(encoded_lines)} lines\n")
        return encoded_lines
    
    def decode(self, idx_seq, drop_padding=False):
        if hasattr(idx_seq, "tolist"):
            idx_seq = idx_seq.tolist()
        else:
            idx_seq = list(idx_seq)

        return ''.join(
            self.idx2token.get(idx, self.oov_token)
            for idx in idx_seq
            if not drop_padding or idx != self.pad_token_idx
        )

class TransformerDataset(Dataset):
    def __init__(self, encoded_lines, context_len,
                 start_token_idx, end_token_idx, pad_token_idx):
        assert len(encoded_lines) >= 1 and context_len >= 1
        assert all(
            len(line) > 2 and
            line[0] == start_token_idx and
            line[-1] == end_token_idx
            for line in encoded_lines
        )

        random.shuffle(encoded_lines)
        self.flattened = [token_idx for line in encoded_lines for token_idx in line]
        self.context_len = context_len


        '''self.lines = encoded_lines
        self.context_len = context_len
        self.pad_token_idx = pad_token_idx
        counts = [len(line)-1 for line in self.lines]
        self.cumulative_counts = list(accumulate(counts))'''

    def __getitem__(self, idx):
        assert (idx >=0 and idx < len(self))
        x = self.flattened[idx: idx + self.context_len]
        y = self.flattened[idx+1: idx + self.context_len+1]
        return torch.tensor(x), torch.tensor(y)


        '''assert (idx >=0 and idx < self.cumulative_counts[-1])

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
        return torch.tensor(x), torch.tensor(y)'''
        

    def __len__(self):
        return len(self.flattened) - self.context_len
        #return self.cumulative_counts[-1]

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

