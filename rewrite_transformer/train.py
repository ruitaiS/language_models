import os
import time

import data
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from model import LanguageModel
import utils

# Transformer Parameters:
context_len = 128
embedding_dim = 512
num_layers = 6
total_heads = 8

# TODO: Use these (rn they're defined within the LM itself)
ffn_expansion_ratio = 4
dropout_params = {
    "embedding_dropout":0.1,
    "post_mha_dropout": 0.1,
    "post_ffn_dropout": 0.1,
    "attention_head_dropout": 0.1,
}

# Data Parameters:
batch_size = 192
validation_p = 0.1
shuffle=True
drop_last=True
tokenization_method='char'
include_book=True

# Training Parameters:
lr=1e-4 * (batch_size / 64)
max_norm = 1.0
print_interval= 100
validation_interval = 100
#rint_interval = 100 * (64 / batch_size)
weight_decay=0.1
epochs = 5

# Tokenizer Initialization ------------------------------------------------------------------------------------
processed_lines = data.preprocess_akjv(include_book)
tokenizer = data.Tokenizer(method=tokenization_method, initialization_text=processed_lines)
encoded_lines = tokenizer.encode_lines(processed_lines)

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
criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=tokenizer.pad_token_idx)
print(f"Optimizer: {optimizer}")

# Train Loop:
training_batches = len(train_loader)
val_batches = len(val_loader)
start = time.time()
for epoch_number in range(epochs):

    for batch_number, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        flattened_logits = logits.view(batch_size*context_len, tokenizer.vocab_size)
        flattened_targets = targets.view(batch_size*context_len).long()
        loss = criterion(flattened_logits, flattened_targets)
        if (batch_number < 5 or batch_number % print_interval == 0 or batch_number == training_batches-1):
            print(f"\n Epoch {epoch_number+1} / {epochs} || {batch_number} / {training_batches-1} || {(time.time() - start):.3f}s || Loss: {loss :.3f}")
            utils.generate(model, tokenizer)
        if (batch_number % validation_interval == 0 and batch_number != 0):
            # Mini Batch Validation Pass Every 10 Print Intervals:
            model.eval()
            with torch.no_grad():
                v_inputs, v_targets = next(iter(val_loader))
                v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                v_logits = model(v_inputs)
                v_loss = criterion(v_logits.view(batch_size*context_len, tokenizer.vocab_size), v_targets.view(batch_size*context_len).long()).item()
                print(f"Mini-batch Validation Loss: {v_loss :.3f}\n")
            model.train()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
    
    # Full Validation Set Loss at the End:
    model.eval()
    print(f"Validating Loss Over {val_batches} Validation Batches...")
    acc_loss = 0
    with torch.no_grad():
        for v_inputs, v_targets in val_loader:
            v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
            v_logits = model(v_inputs)
            acc_loss += criterion(v_logits.view(batch_size*context_len, tokenizer.vocab_size), v_targets.view(batch_size*context_len).long()).item()
    mean_val_loss = acc_loss / val_batches
    print(f"Mean Validation Loss: {mean_val_loss}")
    model.train()


elapsed = time.time() - start
print(f"\nTotal Elapsed time: {elapsed}")
print(f"Batch Size: {batch_size} || LR: {lr}")
print(f"Sequence (Context) Length: {context_len}")
print(f"Embedding Dimension: {embedding_dim}")
print(f"Layers: {num_layers}")
print(f"Heads: {total_heads}\n")