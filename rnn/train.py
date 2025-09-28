import os
import json
import torch
from torch import nn
import rnn
import utils

# parameters ----------------------------------------------
# model:
embedding_dim = 32
hidden_dim = 512
lstm_layers = 3
embedding_dropout = 0.3
lstm_dropout = 0.5
fc_dropout = 0.5
lr = 0.001

# batching:
batch_size = 50
seq_len = 100
validation_p = 0.1
tokenization='char'
include_book=True
shuffle = True

# training:
reset_each = 'batch' # epoch
clip_grad=5
epochs = 30
resume_from = 20
use_gpu = False # TODO: check via code

# chunk data into batches ----------------------------------------------------------
df, full_text_str = utils.preprocess_akjv(include_book)
vocab, vocab_size, idx2token, token2idx, encoded_text_arr = utils.tokenize_str(full_text_str, tokenization)
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


def save_params():
    # Saving Training Params:
    params = {
            'vocab_size': model.vocab_size,
            'tokenization': tokenization,
            'include_book': include_book,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'lstm_layers': model.lstm_layers,
            'embedding_dropout': model.embedding_dropout,
            'lstm_dropout': model.lstm_dropout,
            'fc_dropout': model.fc_dropout,
            'lr': optimizer.param_groups[0]['lr'],
            'batch_size': batch_size,
            'seq_len': seq_len,
            'validation_p': validation_p,
            'shuffle': shuffle,
            'reset_each': reset_each,
            'clip_grad': clip_grad,
            'epochs': epochs,
            'use_gpu': use_gpu
            }

    filename = f"epochs_{resume_from}_to_{resume_from+epochs}.params"
    with open(os.path.join('__checkpoints', 'training_metadata', filename), "w") as f:
        json.dump(params, f, indent=4)


criterion = nn.CrossEntropyLoss()
if resume_from == 0:
    model = rnn.CharRNN(vocab_size, idx2token, token2idx,
                    embedding_dim, hidden_dim, lstm_layers,
                    embedding_dropout, lstm_dropout, fc_dropout, lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    print(f"Model: {model}")
    save_params()
    rnn.train(model, optimizer, criterion, train_loader, val_loader, epochs, reset_each, clip_grad)

else:
    filepath = os.path.join('__checkpoints', f'epoch_{resume_from}.net')
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
    save_params()
    rnn.train(model, optimizer, criterion, train_loader, val_loader, epochs, reset_each, clip_grad, resume_from=resume_from)
