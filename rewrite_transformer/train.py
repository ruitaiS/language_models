import os
import time

import data
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from model import LanguageModel
import generate

def calculate_loss(logits, targets, pad_token_idx=3):
    # In each batch:
    # logits shape (batch_size, seq_len, vocab_size)
    # targets shape (batch_size, seq_len)
    # Think of these as two batch_size, seq_len matrices
    # logits contains a vocab length logits vector in each entry
    # while targets simply contains a token index
    # The logits vectors from one matrix are trying to predict the corresponding token indices in the other.

    # For nn.CrossEntropyLoss:
    # We want to unroll these into two batch_size * seq_len lists, so that:
    # Each element of flattened_logits is a logits vector
    # Each element of flattened_targets is a token index
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=pad_token_idx)
    flattened_logits = logits.view(-1, logits.shape[-1])
    flattened_targets = targets.view(-1)
    return loss_func(flattened_logits, flattened_targets)

def train_model(model_name,
                batch_size,
                context_len,
                embedding_dim,
                num_layers,
                total_heads):
    
    # Pulled from LanguageModel forward pass
    '''
            if targets is not None:
                # View Tokens flowing through:
                #for seq in targets:
                #    view = [self.idx2token[token.item()] for token in seq]
                #    print(' '.join(view))
                #print('')

            #    print(f"Targets Shape: {targets.shape}")
                #targets = np.vectorize(lambda token: self.token2idx.get(token, self.token2idx['<?>']))(targets)
                #targets = torch.tensor(targets, dtype=torch.long)
                loss = self.calculate_loss(logits, targets)
            else:
                loss = None
            return logits, loss
    '''


    # TODO
    #dataset = data.get_dataset('train', dataset_id)
    #aux_data = data.extract_aux(dataset)

    pad_token_idx, idx2token, token2idx = None # TODO

    model = LanguageModel( vocab_size, pad_token_idx, context_len, embedding_dim, num_layers, total_heads)
    model.train()

    print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
    for p in model.parameters():
        p.requires_grad_(True)
        #p.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    print(f"Optimizer: {optimizer}")

    input_batches, target_batches = data.batch(batch_size, context_len, dataset)
    total_batches = len(input_batches)
    print_interval = 500

    try:
        # Training Loss Info
        #with open(os.path.join(base_path, f'models/{model_name}_tr.txt'), 'w') as f:
        #    writer = csv.writer(f, delimiter=' ')

            #start = time.time()
            for batch_index, (X, Y) in enumerate(zip(input_batches, target_batches)):
                logits = model(X)
                loss = calculate_loss(X, Y, pad_token_idx)
                optimizer.zero_grad(set_to_none=True)
                #optimizer.grad.zero_()
                loss.backward()
                #print("Optimizer grad: ", optimizer.grad)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                #elapsed = time.time() - start
                #writer.writerow([batch_index, loss.item(), elapsed])
                if batch_index % print_interval == 1:
                    #seconds = (elapsed / batch_index)*(total_batches - batch_index)
                    #minutes = int(seconds/60)
                    #seconds = int(seconds % 60)
                    #print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}. Estimated time remaining: {minutes}m{seconds}s")
                    print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}.")
    except KeyboardInterrupt:
        #f.close()
        torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
        with open(os.path.join(base_path, f"models/{model_name}_params.json"), 'w') as f:
            json.dump(model_params, f, indent=4)

    f.close()
    torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
    with open(os.path.join(base_path, f"models/{model_name}_params.json"), 'w') as f:
        json.dump(model_params, f, indent=4)

    return model


# Old Transformer Params:
'''train_model(model_name = 'trigram_imitation'
dataset_id = 1741140596
batch_size = 16
context_len = 3
embedding_depth = 8
num_layers = 6
total_heads = 1
masked = True)'''

#RNN Params Copied from rnn/train.py:
'''
# parameters ----------------------------------------------
# model:
embedding_dim = 64
hidden_dim = 512
lstm_layers = 3
embedding_dropout = 0.15
lstm_dropout = 0.3
fc_dropout = 0.3
lr = 0.001
#betas=(0.9, 0.95)
#weight_decay=0.01

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
clip_grad=1
epochs = 30
resume_from = 20
use_gpu = False
'''

# Transformer Parameters:
context_len = 100
embedding_dim = 8
num_layers = 6
total_heads = 1

# Data Parameters:
batch_size = 500
validation_p = 0.1
shuffle=True
drop_last=True
tokenization_method='char'
include_book=True

# Training Parameters:
lr=5e-5
weight_decay=0.01
print_interval=500

# Tokenizer Initialization ------------------------------------------------------------------------------------
processed_lines = data.preprocess_akjv(include_book)
tokenizer = data.Tokenizer(method=tokenization_method, initialization_text=processed_lines)
encoded_lines = tokenizer.encode_lines(processed_lines)

# TODO: Put all this inside a make_loader function
def make_loader(lines, **kwargs):
    pass
# Loader Initialization ------------------------------------------------------------------------------------
akjv_dataset = data.TransformerDataset(encoded_lines, context_len,
                                       start_token_idx=tokenizer.start_token_idx,
                                       end_token_idx=tokenizer.end_token_idx,
                                       pad_token_idx=tokenizer.pad_token_idx)
val_size = int(len(akjv_dataset) * validation_p)
train_size = len(akjv_dataset) - val_size
train_set, val_set = torch.utils.data.random_split(akjv_dataset, [train_size, val_size])

print(f"Dataset Total Size: {len(akjv_dataset)} || Validation Proportion: {validation_p}")
print(f"Training Set Size: {len(train_set)}")
print(f"Validation Set Size: {len(val_set)}")
print(f"Sum: {len(train_set) + len(val_set)}\n")

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last)

print(f"Batch Size: {batch_size} || Drop Last Incomplete Batch: {drop_last}")
print(f"Context Length: {context_len}")
print(f"Train Loader Size: {len(train_loader)} || Instances: {batch_size * len(train_loader)}")
print(f"Validation Loader Size: {len(val_loader)} || Instances: {batch_size * len(val_loader)}\n")

x, y = next(iter(train_loader)) # Remember x, y here are batches
print(f"Sample x.shape: {x.shape}")
print(f"Sample y.shape: {y.shape}\n")
print(f"x[0]:{x[0]}\n")
print(f"y[0]:{y[0]}\n")
print(f"x[0] Reconstructed (Bracketed, Padding Stripped):")
print(f"[{tokenizer.decode(x[0], drop_padding=True)}]\n")
print(f"y[0] Reconstructed (Bracketed, Padding Stripped):")
print(f"[{tokenizer.decode(y[0], drop_padding=True)}]\n")
# --------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device= torch.device("cpu")
print(f"Device: {device}\n")

model = LanguageModel(context_len,
                      embedding_dim,
                      num_layers,
                      total_heads,
                      vocab_size=tokenizer.vocab_size,
                      pad_token_idx=tokenizer.pad_token_idx)
model.to(device)
model.train()

print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
for p in model.parameters():
    p.requires_grad_(True)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
print(f"Optimizer: {optimizer}")

# Single Test Pass with Sample x, y
#logits_batch = model(x)
#print(f"Logits.shape: {logits_batch.shape}\n")
#next_token_batch = generate.sample(logits_batch)
#print(f"next_token_batch.shape: {next_token_batch.shape} || {next_token_batch}")
#print(f"Decoded: {[(idx_seq.tolist(), tokenizer.decode(idx_seq)) for idx_seq in next_token_batch]}")
#print(f"Targets.shape: {y.shape}")

# Test Loop On One Batch:
x, y = x.to(device), y.to(device)
#torch.autograd.set_detect_anomaly(True)
start = time.time()
for i in range(500):
    logits_batch = model(x)
    loss = calculate_loss(logits_batch, y, tokenizer.pad_token_idx)
    print(f"Iteration {i} || Loss: {loss}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
elapsed = time.time() - start
print(f"Elapsed time: {elapsed}")


'''
try:
    # Training Loss Info
    #with open(os.path.join(base_path, f'models/{model_name}_tr.txt'), 'w') as f:
    #    writer = csv.writer(f, delimiter=' ')

        #start = time.time()
        for batch_index, (X, Y) in enumerate(zip(input_batches, target_batches)):
            logits = model(X)
            loss = calculate_loss(X, Y, pad_token_idx)
            optimizer.zero_grad(set_to_none=True)
            #optimizer.grad.zero_()
            loss.backward()
            #print("Optimizer grad: ", optimizer.grad)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            #elapsed = time.time() - start
            #writer.writerow([batch_index, loss.item(), elapsed])
            if batch_index % print_interval == 1:
                #seconds = (elapsed / batch_index)*(total_batches - batch_index)
                #minutes = int(seconds/60)
                #seconds = int(seconds % 60)
                #print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}. Estimated time remaining: {minutes}m{seconds}s")
                print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}.")
except KeyboardInterrupt:
    pass
'''