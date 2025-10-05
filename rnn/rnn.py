import os
import sys
import json
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from nltk.tokenize import RegexpTokenizer

import utils

class CharRNN(nn.Module):
    def __init__(self, tokenization, vocab_size, idx2token, token2idx, pad_token='<>', embedding_dim=32, hidden_dim=512, lstm_layers=3, embedding_dropout=0.3, lstm_dropout=0.5, fc_dropout=0.5, lr=0.001):
        super().__init__()
        assert tokenization in ('char', 'word'), (
        f"tokenization must be 'word' or 'char', got {tokenization}")
        self.tokenization = tokenization
        self.vocab_size = vocab_size
        self.idx2token = idx2token
        self.token2idx = token2idx
        self.pad_token = pad_token
        # earlier models (<9) don't have pad tokens in their vocab
        self.pad_id = self.token2idx.get(pad_token, None)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers =lstm_layers

        self.embedding_dropout = embedding_dropout
        self.lstm_dropout =lstm_dropout
        self.fc_dropout = fc_dropout
        self.lr = lr

        if embedding_dim > 0:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.dropout1 = nn.Dropout(p=embedding_dropout)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers,
                                 dropout=lstm_dropout, batch_first=True)
        else:
            self.embedding = None
            self.dropout1 = None
            self.lstm = nn.LSTM(vocab_size, hidden_dim, lstm_layers,
                                 dropout=lstm_dropout, batch_first=True)

        self.dropout2 = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        if self.embedding_dim > 0:
            x = self.embedding(x)
            x = self.dropout1(x)
        else:
            x = F.one_hot(x, num_classes=self.vocab_size).float()
        output, hidden = self.lstm(x, hidden)
        output = self.dropout2(output)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        #if device is None:
        #    device = next(self.parameters()).device
        device='cpu'
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

def train(model, optimizer, criterion, train_loader, val_loader, epochs, reset_each='epoch', clip_grad=5, use_gpu=False, resume_from=0):
    os.makedirs('__checkpoints', exist_ok=True)
    files = [f for f in os.listdir('__checkpoints')
             if os.path.isfile(os.path.join('__checkpoints', f)) and not f.endswith('.params')]
    if resume_from==0 and files:
        sys.exit(f"Error: __checkpoints is not empty.")

    model.train()
    if use_gpu:
        model.cuda()

    for e in range(epochs):
        hidden = None
        batch_number = 0
        epoch_losses = []
        for inputs, targets in train_loader:
            #print(f"Inputs.shape: {inputs.shape} || Targets.shape: {targets.shape}")
            batch_number += 1
            model.zero_grad()
            if hidden == None or reset_each=='batch':
                hidden = model.init_hidden(inputs.shape[0])
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
            epoch_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            print("Epoch: {}/{} ||".format(resume_from+e+1, resume_from+epochs),
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

        if model.tokenization == 'char':
            # Upper version for whole encoded_text style
            # Lower version for encoded_lines style (stop on padding)
            text = sample(model, stop_token='\n', prime='\t', top_k=None)
            #text = sample(model, stop_token='<>', prime='\t', top_k=None)
        else:
            text = sample(model, stop_token='</s>', prime='<tab>', top_k=None)
            #text = sample(model, stop_token='<>', prime='<tab>', top_k=None)
        print(f"Output Sample: {text}")
        model.train()

        filepath = os.path.join('__checkpoints', f'epoch_{resume_from+e+1}.net')
        print(f"Saving Checkpoint: {filepath}")
        save_rnn_model(model, optimizer, filepath, epoch_losses, val_losses)
    print("Training Complete.")

def save_rnn_model(model, optimizer, filepath, epoch_losses=[], val_losses=[]):
    torch.save({
        'tokenization': model.tokenization,
        'vocab_size': model.vocab_size,
        'idx2token': model.idx2token,
        'token2idx': model.token2idx,
        'pad_token': model.pad_token,
        'pad_id': model.pad_id,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'lstm_layers': model.lstm_layers,
        'embedding_dropout': model.embedding_dropout,
        'lstm_dropout': model.lstm_dropout,
        'fc_dropout': model.fc_dropout,
        'lr': optimizer.param_groups[0]['lr'],
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
        }, filepath)

    # metadata
    path, filename = os.path.split(filepath)
    base, _ = os.path.splitext(filename)
    meta_filename = base + ".meta"
    metadata = {
        # Training Summary:
        'batches_trained': len(epoch_losses),
        'epoch_losses': epoch_losses,
        'val_loss': sum(val_losses) / len(val_losses),
    }
    os.makedirs(os.path.join(path, 'training_metadata'), exist_ok=True)
    with open(os.path.join(path, 'training_metadata', meta_filename), "w") as f:
        json.dump(metadata, f, indent=4)

def load_rnn_model(filepath, optimizer=None):
    checkpoint = torch.load(filepath, map_location="cpu")
    model = CharRNN(
        tokenization=checkpoint.get('tokenization', 'char'),
        vocab_size=checkpoint['vocab_size'],
        idx2token=checkpoint['idx2token'],
        token2idx=checkpoint['token2idx'],
        pad_token=checkpoint.get('pad_token', '<>'),
        embedding_dim=checkpoint.get('embedding_dim', 0),
        hidden_dim=checkpoint['hidden_dim'],
        lstm_layers=checkpoint['lstm_layers'],
        embedding_dropout=checkpoint.get('embedding_dropout', 0),
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

    if top_k is not None and 0 < top_k < logits.size(-1):
        vals, idxs = torch.topk(logits, k=top_k, dim=-1)   # (1,k)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(1, idxs, vals)
        logits = mask
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

def sample(model, stop_token='\n', response_length=None, prime='\n', top_k=None, temperature=1.0):

    # model.cuda()
    model.cpu()
    model.eval()
    if model.tokenization == 'char':
        delimiter = ''
        idx2token = model.idx2token
        priming_indices = [model.token2idx[char] for char in prime]
    else:
        delimiter = ' '
        replacements = {
                '<s>': '',
                '</s>': '\n',
                '<tab>': '    ',
                # '<?>' just leave as-is
                }
        idx2token = {
            idx: replacements.get(tok, tok)
            for idx, tok in model.idx2token.items()
        }
        tokenizer = RegexpTokenizer(r"<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
        words = tokenizer.tokenize(prime)
        priming_indices = [model.token2idx.get(word, model.token2idx['<?>']) for word in words]
    print(f"Priming Tokens: {[model.idx2token[idx] for idx in priming_indices]}")
    print(f"Priming Indices: {priming_indices}")
    hidden = model.init_hidden(batch_size = 1)
    # Iterate over priming chars to build up hidden state
    for token_idx in priming_indices:
        next_idx, hidden = next_token_idx(model, token_idx, hidden, top_k, temperature)

    # Start generating response:
    response_indices = [next_idx]
    if stop_token:
        while response_indices[-1] != model.token2idx[stop_token] and len(response_indices) < 500:
            last_idx = response_indices[-1]
            next_idx, hidden = next_token_idx(model, last_idx, hidden, top_k, temperature)
            response_indices.append(next_idx)
    else:
        assert response_length is not None
        for _ in range(response_length):
            last_idx = response_indices[-1]
            next_idx, hidden = next_token_idx(model, last_idx, hidden, top_k, temperature)
            response_indices.append(next_idx)

    return delimiter.join([idx2token[idx] for idx in response_indices])
