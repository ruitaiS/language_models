import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import utils

class CharRNN(nn.Module):
    def __init__(self, vocab_size, idx2token, token2idx, hidden_dim=512, lstm_layers=2, lstm_dropout=0.5, fc_dropout=0.5, lr=0.001):
        super().__init__()
        self.vocab_size = vocab_size
        self.idx2token = idx2token
        self.token2idx = token2idx
        self.hidden_dim =hidden_dim
        self.lstm_layers =lstm_layers
        self.lstm_dropout =lstm_dropout
        self.fc_dropout = fc_dropout
        self.lr = lr

        self.lstm = nn.LSTM(vocab_size, hidden_dim, lstm_layers,
                             dropout=lstm_dropout, batch_first=True)
        self.final_dropout = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        h, c = hidden
        #print(f"Input x.shape: {x.shape} || h.shape: {h.shape} || cell c.shape: {c.shape}")
        #print(f"One-hot Encoded x.shape: {x.shape}")
        output, (h, c) = self.lstm(x, hidden)
        #print(f"output.shape: {output.shape} || output h.shape: {h.shape} || output c.shape: {c.shape}")

        output = self.final_dropout(output)
        logits = self.fc(output)
        #print(f"Final logits.shape: {logits.shape}")
        return logits, (h, c)

    def init_hidden(self, batch_size):
        #if device is None:
        #    device = next(self.parameters()).device
        device='cpu'
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

def train(model, optimizer, criterion, train_loader, val_loader, epochs, clip_grad=5, use_gpu=False):
    os.makedirs('checkpoints', exist_ok=True)
    model.train()
    if use_gpu:
        model.cuda()

    for e in range(epochs):
        # TODO: not sure about resetting only once per epoch
        hidden = None
        batch_number = 0
        for inputs, targets in train_loader:
            #print(f"Inputs.shape: {inputs.shape} || Targets.shape: {targets.shape}")
            batch_number += 1
            model.zero_grad()
            if hidden == None:
                hidden = model.init_hidden(inputs.shape[0])
            #hidden = model.init_hidden(inputs.shape[0])
            h, c = hidden
            hidden = (h.detach(), c.detach())
            logits, hidden = model(inputs, hidden)
            # TODO: More elegant way of retrieving these dims
            batch_size, seq_len, vocab_size = logits.shape
            flattened_logits = logits.view(batch_size*seq_len, vocab_size)
            flattened_targets = targets.view(batch_size*seq_len).long()
            #print(f"Flattened logits.shape: {flattened_logits.shape}")
            #print(f"Flattened targets.shape: {flattened_targets.shape}")
            loss = criterion(flattened_logits, flattened_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            print("Epoch: {}/{} ||".format(e+1, epochs),
                  "Batch Number: {}/{} ||".format(batch_number, len(train_loader)),
                  "Batch Loss: {:.4f}".format(loss.item())
                  )

        # Every Epoch, check loss on entire validation set:
        print("Calculating Validation Set Loss...")
        val_hidden = None
        val_losses = []
        model.eval()
        for inputs, targets in val_loader:
            if val_hidden == None:
                val_hidden = model.init_hidden(inputs.shape[0])
            v_h, v_c = val_hidden
            val_hidden = (v_h.detach(), v_c.detach())
            val_logits, hidden = model(inputs, val_hidden)
            # TODO: More elegant way of retrieving these dims
            batch_size, seq_len, vocab_size = val_logits.shape
            flattened_vlogits = val_logits.view(batch_size*seq_len, vocab_size)
            flattened_vtargets = targets.view(batch_size*seq_len).long()
            #print(f"Flattened logits.shape: {flattened_vlogits.shape}")
            #print(f"Flattened targets.shape: {flattened_vtargets.shape}")
            val_loss = criterion(flattened_vlogits, flattened_vtargets)
            val_losses.append(val_loss.item())
        val_loss_mean = sum(val_losses) / len(val_losses)
        print("Validation Set Loss: {:.4f}".format(val_loss_mean))

        text = sample(model, response_length=100, prime='Genesis', top_k=None)
        print(f"100 Char Sample: {text}")
        model.train()

        filepath = os.path.join('checkpoints', f'epoch_{e+1}.net')
        print(f"Saving Checkpoint: {filepath}")
        save_rnn_model(model, optimizer, filepath)
    print("Training Complete.")

def save_rnn_model(model, optimizer, filepath):
    torch.save({
        'vocab_size': model.vocab_size,
        'idx2token': model.idx2token,
        'token2idx': model.token2idx,
        'hidden_dim': model.hidden_dim,
        'lstm_layers': model.lstm_layers,
        'lstm_dropout': model.lstm_dropout,
        'fc_dropout': model.fc_dropout,
        'lr': model.lr,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
        }, filepath)

def load_rnn_model(filepath, optimizer=None):
    checkpoint = torch.load(filepath, map_location="cpu")
    model = CharRNN(
        vocab_size=checkpoint['vocab_size'],
        idx2token=checkpoint['idx2token'],
        token2idx=checkpoint['token2idx'],
        hidden_dim=checkpoint['hidden_dim'],
        lstm_layers=checkpoint['lstm_layers'],
        lstm_dropout=checkpoint['lstm_dropout'],
        fc_dropout=checkpoint['fc_dropout'],
        lr=checkpoint['lr']
    )
    model.load_state_dict(checkpoint['state_dict'])
    if not optimizer:
        optimizer=torch.optim.Adam(model.parameters(), lr=model.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return model, optimizer

def next_token_idx(model, token_idx, hidden, top_k=None, temperature=1.0):
    model.eval()
    h, c = hidden
    hidden = (h.detach(), c.detach())
    logits, hidden = model(torch.tensor([[token_idx]]), hidden)
    logits = logits[:, -1, :] / max(temperature, 1e-8)

    probs = F.softmax(logits, dim=1)
    # if gpu:
        # probs = probs.cpu()

    #if top_k is None:
    #    indices = np.arange(model.vocab_size)
    #else:
    #    probs, indices = probs.topk(top_k)
    #    indices = indices.numpy().squeeze()

    #probs = probs.numpy().squeeze()
    #next_idx = np.random.choice(indices, p = probs/probs.sum())
    next_idx = torch.multinomial(probs, num_samples=1).item()
    return next_idx, hidden

def sample(model, response_length, prime='\n', top_k=None, temperature=1.0):
    # model.cuda()
    model.cpu()
    model.eval()
    priming_indices = [model.token2idx[char] for char in prime]
    hidden = model.init_hidden(batch_size = 1)
    # Iterate over priming chars to build up hidden state
    for token_idx in priming_indices:
        next_idx, hidden = next_token_idx(model, token_idx, hidden, top_k, temperature)

    # Start generating response:
    response_indices = [next_idx]
    for _ in range(response_length):
        last_idx = response_indices[-1]
        next_idx, hidden = next_token_idx(model, last_idx, hidden, top_k, temperature)
        response_indices.append(next_idx)

    return ''.join([model.idx2token[idx] for idx in response_indices])



# hyperparameters
batch_size = 100
seq_len = 100
epochs = 10
validation_p = 0.1
use_gpu = False # TODO: check via code

df, full_text_str = utils.preprocess_akjv()
vocab, vocab_size, idx2token, token2idx, encoded_text_arr = utils.tokenize_str(full_text_str)
train_loader, val_loader = utils.make_dataloader(encoded_text_arr,
                                           batch_size=batch_size,
                                           seq_len=seq_len,
                                           validation_p=validation_p,
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
retrain = False
if retrain:
    model = CharRNN(vocab_size=vocab_size, idx2token=idx2token, token2idx=token2idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    print(f"Model: {model}")
    train(model, optimizer, criterion, train_loader, val_loader, epochs, use_gpu=False)
else:
    filepath = os.path.join('checkpoints', 'epoch_10.net')
    model, optimizer = load_rnn_model(filepath)

    text = sample(model, response_length=100, prime='Genesis', top_k=None)
    print(f"100 Char Sample: {text}")

    #optimizer.lr = 0.0005
    #train(model, optimizer, criterion, train_loader, val_loader, epochs, use_gpu=False)
    print(f"Model: {model}")
