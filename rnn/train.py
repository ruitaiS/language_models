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
style='encoded_lines'
pad_token='<>'

# training:
reset_each = 'batch' # epoch
clip_grad=5
epochs = 20
resume_from = 0
use_gpu = False

df, full_text_str = utils.preprocess_akjv(include_book)
# Note vocab is built from entire text corpus (train/val data leakage)
# For our purposes i think its ok
vocab, vocab_size, idx2token, token2idx, encoded_text, encoded_lines = utils.tokenize(df, full_text_str, tokenization, pad_token=pad_token)


if style=='encoded_text':
    encoded_arr = encoded_text
else:
    encoded_arr = encoded_lines

#if tokenization=='char':
#    eol_idx = token2idx.get('\n', None)
#else:
#    eol_idx = token2idx.get('</s>', None)

pad_idx = token2idx.get(pad_token, None)
train_loader, val_loader = utils.make_dataloader(encoded_arr,
                                                 batch_size=batch_size,
                                                 seq_len=seq_len,
                                                 validation_p=validation_p,
                                                 shuffle=shuffle,
                                                 style=style,
                                                 #eol_idx = eol_idx,
                                                 pad_idx=pad_idx)

x, y = next(iter(train_loader))
print('')
print(df.head(3))
print('\n' + full_text_str[:273])
print(encoded_text[:273])
print(f"\nx.shape: {x.shape}")
print(f"y.shape: {y.shape}")
print('\ntruncated x =\n', x[:10, :10])
print('\ntruncated y =\n', y[:10, :10])


def save_params():
    # Saving Training Params:
    params = {
            'pad_token': model.pad_token,
            'pad_id': model.pad_id,
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
    os.makedirs(os.path.join('__checkpoints', 'training_metadata'), exist_ok=True)
    with open(os.path.join('__checkpoints', 'training_metadata', filename), "w") as f:
        json.dump(params, f, indent=4)


criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
if resume_from == 0:
    model = rnn.CharRNN(tokenization, vocab_size, idx2token, token2idx, pad_token,
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


    if model.tokenization == 'char':
        prime = '\t'
        stop_char='\n'
    else:
        prime = '<tab>'
        stop_char = '</s>'

    #text = sample(model, stop_char='\n', prime='Genesis\t', temperature=0.65)
    text = rnn.sample(model, stop_char=stop_char, prime=prime, temperature=1.0)
    print(f"\n{text}")
    text = rnn.sample(model, stop_char=stop_char, prime=prime, temperature=1.0)
    print(f"\n{text}")
    text = rnn.sample(model, stop_char=stop_char, prime=prime, temperature=1.0)
    print(f"\n{text}")

    optimizer.lr = 0.00025
    model.lr = 0.00025
    save_params()
    rnn.train(model, optimizer, criterion, train_loader, val_loader, epochs, reset_each, clip_grad, resume_from=resume_from)
