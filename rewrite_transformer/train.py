import os
import time

import data
from data import Config
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from model import LanguageModel
import utils

# TODO: load_cfg logic here

default = {
    # Transformer Parameters:
    'context_len': 512, # GPT2: 1024
    'embedding_dim': 768,
    'num_layers': 12,
    'heads_per_layer': 12,
    'ffn_expansion_ratio': 4, # 768 * 4 = 3072 FFN Hidden Dim

    # GPT2 uses no dropout btw!
    "embedding_dropout":0.1,
    "post_mha_dropout": 0.1,
    "post_ffn_dropout": 0.1,
    "attention_head_dropout": 0.1,

    # Data Parameters:
    'tk_method': 'word', # GPT2: BPE
    'include_book': True,

    'batch_size': 16, # GPT2: 512
    'validation_p': 0.1,
    'shuffle': True,
    'drop_last': True,
    'num_workers': 4, # or 8
    'pin_memory': True,
    'prefetch_factor': 2, # to 4
    'persistent_workers': True,

    # Training Parameters:
    'lr': 5e-4 * (16 / 512), #  batch_size/512
    'max_norm': 1.0,
    'print_interval': 100,
    'validation_interval': 100,
    'weight_decay': 0.1,
    'epochs': 5,
}

cfg = Config(default)


# Tokenizer Initialization ------------------------------------------------------------------------------------
processed_lines = data.preprocess_akjv(cfg.include_book)
tokenizer = data.Tokenizer(method=cfg.tk_method, init_text=processed_lines)
encoded_lines = tokenizer.encode_lines(processed_lines)
cfg.tokenizer = tokenizer.cfg()

# Loader Initialization ------------------------------------------------------------------------------------
print(f"Context Length: {cfg.context_len}")
print(f"Batch Size: {cfg.batch_size} || Drop Last Incomplete Batch: {cfg.drop_last}")
akjv_dataset = data.TransformerDataset(encoded_lines, cfg.context_len,
                                       start_token_idx=cfg.tokenizer.start_token_idx,
                                       end_token_idx=cfg.tokenizer.end_token_idx,
                                       pad_token_idx=cfg.tokenizer.pad_token_idx)
train_loader, val_loader = data.build_train_val_loaders(akjv_dataset, cfg.batch_size, cfg.validation_p)
# --------------------------------------------------------------------------------------------------------------

# Model, Optimizer, and Loss Function
model = LanguageModel(cfg)
for p in model.parameters():
    p.requires_grad_(True)
optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=cfg.tokenizer.pad_token_idx)
print(f"Total Model Parameters: {sum(p.nelement() for p in model.parameters())}")
print(f"Optimizer: {optimizer}")
print(f"Loss Function: {criterion}")
print(f"Learning Rate: {cfg.lr}")

# Instantiating Training Loop Variables:
if cfg.device is None:
    cfg.device = "cuda" # "cpu"
device = torch.device(cfg.device)
epochs = cfg.epochs
training_batches = len(train_loader)
val_batches = len(val_loader)
batch_size = cfg.batch_size
context_len = cfg.context_len
vocab_size = cfg.tokenizer.vocab_size
print_interval = cfg.print_interval
validation_interval = cfg.validation_interval
max_norm = cfg.max_norm

# Train Start
model.to(device)
print(f"Device: {model.device}\n")
print(f"Printing Updates Every {cfg.print_interval} batches || {cfg.print_interval * cfg.batch_size } sequences\n")
model.train()
start = time.time()
for epoch_number in range(epochs):
    for batch_number, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        flattened_logits = logits.view(batch_size*context_len, vocab_size)
        flattened_targets = targets.view(batch_size*context_len).long()
        loss = criterion(flattened_logits, flattened_targets)
        if (batch_number % print_interval == 1 or batch_number == training_batches-1):
        #if (batch_number % print_interval == 0 or batch_number == training_batches-1):
            elapsed = time.time() - start
            estimated = elapsed * (training_batches-1)/(batch_number)
            remaining = estimated * epochs - elapsed
            h, hr = int(estimated // 3600), int(remaining // 3600)
            m, mr = int((estimated % 3600) // 60), int((remaining % 3600) // 60)
            s, sr = estimated % 60, remaining % 60
            
            print(f"\n Epoch {epoch_number+1} / {epochs} || {batch_number} / {training_batches-1} || {(time.time() - start):.3f}s || Loss: {loss :.3f}")
            print(f"Estimated Time Per Epoch: {h}h:{m}m:{s:.2f}s || Estimated Time Remaining: {hr}h:{mr}m:{sr:.2f}s\n")
            utils.generate(model, tokenizer)
        if (batch_number % validation_interval == 0 and batch_number != 0):
            # Mini Batch Validation Pass Every 10 Print Intervals:
            model.eval()
            with torch.no_grad():
                v_inputs, v_targets = next(iter(val_loader))
                v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                v_logits = model(v_inputs)
                v_loss = criterion(v_logits.view(batch_size*context_len, vocab_size), v_targets.view(batch_size*context_len).long()).item()
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
            acc_loss += criterion(v_logits.view(batch_size*context_len, vocab_size), v_targets.view(batch_size*context_len).long()).item()
    mean_val_loss = acc_loss / val_batches
    print(f"Mean Validation Loss: {mean_val_loss}")
    model.train()


elapsed = time.time() - start
print(f"\nTotal Elapsed time: {elapsed}")
print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
print(f"Batch Size: {batch_size} || Learning Rate: {cfg.lr}")
print(f"Context Length: {context_len}")
print(f"Embedding Dimension: {model.embedding_dim}")
print(f"Layers: {model.num_layers}")
print(f"Heads per Layer: {model.heads_per_layer}\n")

# TODO:
# Load CFG
# Save CFG
# Save Model (+ Handle interrupts)
# Resume From Saved
