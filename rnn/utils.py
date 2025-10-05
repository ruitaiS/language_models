import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer

def tokenize(df, full_text_str, tokenization='char', pad_token='<>'):
    assert tokenization in ('char', 'word'), (
    f"tokenization must be 'word' or 'char', got {tokenization}")
    if tokenization=='char':
        # Note for character level tokenization, `\n` counts as one character
        # and forms a natural "end of line" character!
        # `\t` likewise counts as one character,
        # and forms a natural 'book' vs. 'text' delineator in the akjv.txt corpus
        #idx2token = dict(enumerate(vocab))
        #token2idx = {ch:ii for ii, ch in idx2token.items()}
        print("Char Tokenizaiton")
        vocab = [pad_token] + sorted(tuple(set(full_text_str)))
        vocab_size = len(vocab)
        idx2token =  vocab
        token2idx = {token: idx for idx, token in enumerate(idx2token)}
        encoded_text = np.array([token2idx[ch] for ch in full_text_str], dtype=np.int32)
        encoded_lines = [
                [token2idx[ch] for ch in text]
                for text in df['text']
                ]
    else:
        full_text_str = "<s> " + full_text_str.replace("\n", " </s> <s> ")
        full_text_str = full_text_str.replace("\t", " <tab> ")
        cut_chars = len(" <s> ")
        print(f"Cutting Before Tokenizing: {full_text_str[-cut_chars:]!r}")
        full_text_str = full_text_str[:-len(" <s> ")]
        #tokenizer = RegexpTokenizer(r"<[^>\s]+>|\w+|[^\w\s]")
        tokenizer = RegexpTokenizer(r"<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
        words = tokenizer.tokenize(full_text_str)
        vocab = sorted(set(words))
        vocab.insert(0, pad_token)
        vocab.insert(0, '<?>') # unseen word token
        vocab_size = len(vocab)
        idx2token = dict(enumerate(vocab))
        token2idx = {word:i for i, word in idx2token.items()}
        encoded_text = np.array([token2idx[word] for word in words], dtype=np.int32)
        encoded_lines = [
                [token2idx[token] for token in tokenizer.tokenize(text)]
                for text in df['text']
                ]
    print(f"Vocabulary Size: {vocab_size}")
    return vocab, vocab_size, idx2token, token2idx, encoded_text, encoded_lines

def preprocess_akjv(include_book=True):
    source_filepath = os.path.join('..', 'datasets', 'akjv.txt')
    # utf-8-sig strips leading BOM char
    with open(source_filepath, encoding="utf-8-sig") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    s = pd.DataFrame(lines, columns=["raw"])
    df = s["raw"].str.extract(
        r'^\s*(?P<book>.+?)\s+(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.+?)\s*$',
        expand=True
    )
    df = df.dropna().reset_index(drop=True)
    if include_book:
        #full_text_str = "\n".join(df["book"].astype(str) + "\t" + df["text"].astype(str))
        lines = "<" + df["book"].astype(str) + ">"  + "\t" + df["text"].astype(str)
        full_text_str = "\n".join(lines)
    else:
        full_text_str = "\n".join("\t" + s for s in df["text"].astype(str))
    full_text_str = full_text_str + '\n' # Ensure last line also gets `end_of_line` character
    return df, full_text_str

class RnnDataset(Dataset):
    def __init__(self, arr, seq_len, style='encoded_text', pad_idx=None):
        assert style in ('encoded_text', 'encoded_lines'), (
                f'style must be "encoded_text" or "encoded_lines"')
        self.style = style
        self.seq_len = seq_len
        self.pad_idx = pad_idx

        if style == 'encoded_text':
            # keep only complete input sequences x
            # reserve one index position for y = x + 1 shift
            keep_len = ((len(arr) - 1) // seq_len) * seq_len
            self.arr = arr[:keep_len+1]
            self.len = keep_len // seq_len
        else:
            self.arr = arr
            self.len = len(arr)

    def __getitem__(self, idx):
        if self.style == 'encoded_text':
            start = idx * self.seq_len
            end = start + self.seq_len
            x = self.arr[start:end]
            y = self.arr[start+1:end+1]
            return torch.from_numpy(x).long(), torch.from_numpy(y).long()
        else:
            x = self.arr[idx]
            y = np.concatenate((x[1:], [self.pad_idx]))
            print(f"x: {x}")
            print(f"y: {y}")
            return torch.from_numpy(x).long(), torch.from_numpy(y).long()

    def __len__(self):
        return self.len

def pad_lines(encoded_lines, seq_len, pad_idx, shuffle=False):
    res = np.full((len(encoded_lines), seq_len), pad_idx)
    for i, encoded_line in enumerate(encoded_lines):
        truncated = encoded_line[:seq_len]
        res[i, : len(truncated)] = truncated
    if shuffle:
        np.random.shuffle(res)
    return res

def make_dataloader(encoded_arr,
                    batch_size, seq_len,
                    validation_p,
                    shuffle=False,
                    style='encoded_text',
                    #eol_idx=0,
                    pad_idx=None):
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print(f"Validation Proportion: {validation_p}")
    print(f"Encoded Length: {len(encoded_arr)}")

    assert style in ('encoded_text', 'encoded_lines'), (
            f'style must be "encoded_text" or "encoded_lines"')

    if style == 'encoded_text':
        # train/val split at closest batch_size * seq_len to validation_p
        # feels wrong to have validation set be an unshuffled chunk, but ok
        split_idx = (int)((len(encoded_arr) * (1-validation_p))//(batch_size*seq_len))*(batch_size*seq_len)
        print(f"Split Index: {split_idx}")
        assert split_idx + 1 != len(encoded_arr)

    else:
        assert pad_idx is not None
        encoded_arr = pad_lines(encoded_arr, seq_len, pad_idx, shuffle=shuffle)
        split_idx = (int)((len(encoded_arr) * (1-validation_p))//(batch_size))*(batch_size)
        print(f"Split Index: {split_idx}")
        assert split_idx + 1 != len(encoded_arr)

    train_loader = torch.utils.data.DataLoader(
            RnnDataset(encoded_arr[:split_idx+1], seq_len, style=style, pad_idx=pad_idx),
            #RnnDataset(encoded_arr[:split_idx+1], seq_len),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True)
    print(f"Train Loader Size: {len(train_loader)}")

    val_loader = torch.utils.data.DataLoader(
            RnnDataset(encoded_arr[split_idx:], seq_len, style=style, pad_idx=pad_idx),
            #RnnDataset(encoded_arr[split_idx:], seq_len),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True)
    print(f"Validation Loader Size: {len(val_loader)}")
    assert len(val_loader) >= 0

    return train_loader, val_loader
