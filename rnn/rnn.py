import torch
from torch import nn

from utils import one_hot_encode, make_dataloader, preprocess_akjv

class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, p_dropout=0.5, lr=0.001):
        super().__init__()
        self.tokens = tokens
        self.p_dropout = p_dropout
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        #TODO define layers

    def forward(self, x, hidden):
        # TODO
        print('')

    def init_hidden(self, batch_size):
        # TODO
        print('')


# Testing -----
df, full_text_str, encoded_text_arr, vocab, vocab_size, int2word, word2int = preprocess_akjv()
akjv_loader = make_dataloader(encoded_text_arr,
                              batch_size = 8, seq_len = 50,
                              style='RNN')
x, y = next(iter(akjv_loader))
print(df.head())
print(full_text_str[:100])
print(encoded_text_arr[:10])
print('\nsample x =\n', x[:10, :10])
print('\nsample y =\n', y[:10, :10])
