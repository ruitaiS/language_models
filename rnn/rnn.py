import torch
from torch import nn
import torch.nn.functional as F

from utils import one_hot_encode, make_dataloader, preprocess_akjv

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, lstm_layers=2, lstm_dropout=0.5, fc_dropout=0.5, lr=0.001):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm_dropout =lstm_dropout
        self.lstm_layers =lstm_layers
        self.hidden_dim =hidden_dim
        self.lr = lr

        self.lstm = nn.LSTM(vocab_size, hidden_dim, lstm_layers,
                             dropout=lstm_dropout, batch_first=True)
        self.final_dropout = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        #x = one_hot_encode(x, self.vocab_size)
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

def train(model, train_loader, val_loader, epochs, clip_grad=5, eval_every=10, use_gpu=False):
    model.train()
    if use_gpu:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    criterion = nn.CrossEntropyLoss()

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

        #if batch_number % eval_every == 0:
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
        model.train()
    print("Training Complete.")






# hyperparameters
batch_size = 10
seq_len = 50
epochs = 10
validation_p = 0.2
use_gpu = False # TODO: check via code

df, full_text_str, encoded_text_arr, vocab, vocab_size, int2word, word2int = preprocess_akjv()
train_loader, val_loader = make_dataloader(encoded_text_arr,
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

model = CharRNN(vocab_size=vocab_size)
print(f"Model: {model}")

train(model, train_loader, val_loader, epochs, eval_every=10, use_gpu=False)
