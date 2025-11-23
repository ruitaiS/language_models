import os
import time

import data
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from model import LanguageModel
import generate

def calculate_loss(loss_function, logits, targets):
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
    
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_targets = targets.view(-1)

    #print("logits stats: min", logits.min().item(),
    #  "max", logits.max().item(),
    #  "mean", logits.mean().item(),
    #  "std", logits.std().item())
    #print("targets range:", flat_targets.min().item(), "to", flat_targets.max().item())

    return loss_function(flat_logits, flat_targets)


# Transformer Parameters:
context_len = 256
embedding_dim = 512
num_layers = 6
total_heads = 8

# Data Parameters:
batch_size = 192
validation_p = 0.1
shuffle=True
drop_last=True
tokenization_method='char'
include_book=True

# Training Parameters:
epochs = 5
lr=1e-4 * (batch_size / 64)
print_interval= 100
#rint_interval = 100 * (64 / batch_size)
weight_decay=0.1

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

print(f"Learning Rate: {lr}")
print(f"Print Every {print_interval} batches || {print_interval * batch_size } sequences\n")

'''x, y = next(iter(train_loader)) # Remember x, y here are batches
print(f"Sample x.shape: {x.shape}")
print(f"Sample y.shape: {y.shape}\n")
print(f"x[0]:{x[0]}\n")
print(f"y[0]:{y[0]}\n")
print(f"x[0] Reconstructed (Bracketed, Padding Stripped):")
print(f"[{tokenizer.decode(x[0], drop_padding=True)}]\n")
print(f"y[0] Reconstructed (Bracketed, Padding Stripped):")
print(f"[{tokenizer.decode(y[0], drop_padding=True)}]\n")'''
# --------------------------------------------------------------------------------------------------------------

device = torch.device("cuda")
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

# Optimizer + Loss Function Definition
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
print(f"Optimizer: {optimizer}")
loss_func = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=tokenizer.pad_token_idx)

# Train Loop:
training_batches = len(train_loader)
start = time.time()
for i in range(epochs):
    for j, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits_batch = model(x)
        loss = calculate_loss(loss_func, logits_batch, y)
        if (j < 5 or j % print_interval == 0 or j == training_batches-1):
            # TODO: Avg. Loss Over Validation Set
            print(f"\n Epoch {i+1} / {epochs} || {j} / {training_batches-1} || {(time.time() - start):.3f}s || Loss: {loss :.3f}")
            generate.test_generate(model, tokenizer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

elapsed = time.time() - start
print(f"\nTotal Elapsed time: {elapsed}")
print(f"Batch Size: {batch_size} || LR: {lr}")
print(f"Sequence (Context) Length: {context_len}")
print(f"Embedding Dimension: {embedding_dim}")
print(f"Layers: {num_layers}")
print(f"Heads: {total_heads}\n")