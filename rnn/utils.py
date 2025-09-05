import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import pandas as pd

def one_hot_encode(arr, vocab_size):
    # One-hot encode vectors from any dimension integer array
    one_hot = np.zeros((arr.size, vocab_size), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, vocab_size))
    return one_hot

def get_vocab(full_text_str, tokenization='char'):
    # Room to implement different tokenization methods
    # Note for character level tokenization, `\n` counts as one character
    # and forms a natural "end of line" character!
    # `\t` likewise counts as one character,
    # and forms a natural 'book' vs. 'text' delineator in the akjv.txt corpus
    vocab = sorted(tuple(set(full_text_str)))
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size}")
    return vocab, vocab_size

def preprocess_akjv():
    # each source file gets its own preprocess function
    # passed as preprocess_text() param for TextDataset
    # returns df, full_text_str, encoded_text_arr, vocab, vocab_size, int2word, word2int
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
    full_text_str = "\n".join(df["book"].astype(str) + "\t" + df["text"].astype(str))
    full_text_str = full_text_str + '\n' # Ensure last line also gets `end_of_line` character
    vocab, vocab_size = get_vocab(full_text_str)
    # "word" means any element in our vocab (eg. a token)
    # but word just mentally maps more easily to vocab
    int2word = dict(enumerate(vocab))
    word2int = {ch:ii for ii, ch in int2word.items()}
    encoded_text_arr = np.array([word2int[ch] for ch in full_text_str])
    return df, full_text_str, encoded_text_arr, vocab, vocab_size, int2word, word2int

class RnnDataset(Dataset):
    '''
    RNN format is a set of non-overlapping seq_len sized arrays
    Transformer format is a sliding window of seq_len arrays

    you could adopt this code for transformer format w/ tweaks to the indexing logic
    '''
    def __init__(self, arr, seq_len):
        # keep only complete input sequences x
        # reserve one index position for y = x + 1 shift
        keep_len = ((len(arr) - 1) // seq_len) * seq_len
        self.arr = arr[:keep_len+1]
        self.len = keep_len // seq_len
        self.seq_len = seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.arr[start:end]
        y = self.arr[start+1:end+1]
        return x, y

    def __len__(self):
        return self.len

def make_dataloader(encoded_text_arr,
                    batch_size, seq_len, style='RNN'):
    if style == 'RNN':
        return torch.utils.data.DataLoader(
                RnnDataset(encoded_text_arr, seq_len),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True)
    elif style == 'Transformer':
        #shuffle = True
        #drop_last = False # ? or True? idk
        print("TODO")
    else:
        raise ValueError(f"{style} is not a valid style value ('RNN' or 'Transformer')")
