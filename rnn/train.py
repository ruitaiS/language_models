import os
import torch
from torch import nn
import rnn
import utils

# parameters ----------------------------------------------
# model:
hidden_dim = 512
lstm_layers = 3
lstm_dropout = 0.5
fc_dropout = 0.5
lr = 0.001

# batching:
batch_size = 50
seq_len = 100
validation_p = 0.1
include_book=False
shuffle = True

# training:
reset_each = 'batch' # epoch
clip_grad=5
epochs = 20
use_gpu = False # TODO: check via code



# chunk data into batches ----------------------------------------------------------
df, full_text_str = utils.preprocess_akjv(include_book)
vocab, vocab_size, idx2token, token2idx, encoded_text_arr = utils.tokenize_str(full_text_str)
train_loader, val_loader = utils.make_dataloader(encoded_text_arr,
                                                 batch_size=batch_size,
                                                 seq_len=seq_len,
                                                 validation_p=validation_p,
                                                 shuffle=shuffle,
                                                 style='RNN')

x, y = next(iter(train_loader))
print('')
print(df.head(3))
print('\n' + full_text_str[:273])
print(encoded_text_arr[:273])
print(f"\nx.shape: {x.shape}")
print(f"y.shape: {y.shape}")
print('\ntruncated x =\n', x[:10, :10])
print('\ntruncated y =\n', y[:10, :10])


criterion = nn.CrossEntropyLoss()
restart = True
if restart:
    model = rnn.CharRNN(vocab_size, idx2token, token2idx,
                    hidden_dim, lstm_layers,
                    lstm_dropout, fc_dropout, lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    print(f"Model: {model}")
    rnn.train(model, optimizer, criterion, train_loader, val_loader, epochs, reset_each, clip_grad, use_gpu)
else:
    filepath = os.path.join('__checkpoints', 'epoch_30.net')
    model, optimizer = rnn.load_rnn_model(filepath)
    print(f"\nModel: {model}")

    #text = sample(model, stop_char='\n', prime='Genesis\t', temperature=0.65)
    text = rnn.sample(model, stop_char='\n', prime='\t', temperature=1.0)
    print(f"\n{text}")
    text = rnn.sample(model, stop_char='\n', prime='\t', temperature=1.0)
    print(f"\n{text}")
    text = rnn.sample(model, stop_char='\n', prime='\t', temperature=1.0)
    print(f"\n{text}")

    optimizer.lr = 0.00025
    model.lr = 0.00025
    rnn.train(model, optimizer, criterion, train_loader, val_loader, epochs, reset_each, clip_grad, use_gpu)

