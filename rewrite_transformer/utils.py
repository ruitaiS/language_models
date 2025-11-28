import os
import json
import time
import torch
import warnings
from torch.optim import AdamW
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import data
from model import LanguageModel

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # wrap nested dicts at initialization
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = Config(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            warnings.warn(f"[Config] No such config field: '{key}'")
            return None

    def __setattr__(self, key, value):
        # wrap assigned dicts as Config recursively
        if isinstance(value, dict):
            value = Config(value)
        dict.__setitem__(self, key, value)

def init(cfg, verbose=True):
    # Tokenizer Initialization ------------------------------------------------------------------------------------
    processed_lines = data.preprocess_akjv(cfg.include_book)
    tokenizer = data.Tokenizer(method=cfg.tk_method, init_text=processed_lines)
    encoded_lines = tokenizer.encode_lines(processed_lines)
    cfg.tokenizer = tokenizer.cfg()

    # Loader Initialization ------------------------------------------------------------------------------------
    akjv_dataset = data.TransformerDataset(encoded_lines, cfg.context_len,
                                        start_token_idx=cfg.tokenizer.start_token_idx,
                                        end_token_idx=cfg.tokenizer.end_token_idx,
                                        pad_token_idx=cfg.tokenizer.pad_token_idx)
    train_loader, val_loader = data.build_train_val_loaders(akjv_dataset, cfg)
    # --------------------------------------------------------------------------------------------------------------
    # Model, Optimizer, and Loss Function
    model = LanguageModel(cfg)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=cfg.tokenizer.pad_token_idx)
    
    if verbose:
        print(f"Context Length: {cfg.context_len}")
        print(f"Batch Size: {cfg.batch_size} || Drop Last Incomplete Batch: {cfg.drop_last}")
        print(f"Total Model Parameters: {sum(p.nelement() for p in model.parameters())}")
        print(f"Optimizer: {optimizer}")
        print(f"Loss Function: {criterion}")
        print(f"Learning Rate: {cfg.lr}")

    return tokenizer, model, optimizer, criterion, train_loader, val_loader

def train(cfg, tokenizer, model, optimizer, criterion, train_loader, val_loader):

    # Instantiating Training Loop Variables:
    device = torch.device(cfg.device)
    epochs = cfg.epochs
    training_batches = len(train_loader)
    val_batches = len(val_loader)
    batch_size = cfg.batch_size
    context_len = cfg.context_len
    vocab_size = cfg.tokenizer.vocab_size
    print_interval = cfg.print_interval
    max_norm = cfg.max_norm
    #training_log = {} # TODO

    # Train Start
    model.to(device)
    print(f"Device: {model.device}\n")
    print(f"Printing Updates Every {cfg.print_interval} batches || {cfg.print_interval * cfg.batch_size } sequences\n")
    model.train()
    start = time.time()
    for epoch_number in range(epochs):
        train_losses = []
        val_losses = []
        for batch_number, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            flattened_logits = logits.view(batch_size*context_len, vocab_size)
            flattened_targets = targets.view(batch_size*context_len).long()
            loss = criterion(flattened_logits, flattened_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            
            if (batch_number % print_interval == 1 or batch_number == training_batches-1):
                elapsed = time.time() - start
                estimated = elapsed * (training_batches-1)/(batch_number)
                remaining = estimated * (epochs - epoch_number) - elapsed
                h, hr, eh = int(estimated // 3600), int(remaining // 3600), int(elapsed // 3600)
                m, mr, em = int((estimated % 3600) // 60), int((remaining % 3600) // 60), int((elapsed % 3600) // 60)
                s, sr, es = estimated % 60, remaining % 60, elapsed % 60

                train_losses.append(loss.item())    
                print(f"\n Epoch {epoch_number+1} / {epochs} || {batch_number} / {training_batches-1} || {eh}h:{em}m:{es:.2f}s || Loss: {loss :.3f}")
                print(f"Elapsed: {eh}h:{em}m:{es:.2f}s || Estimated Time Per Epoch: {h}h:{m}m:{s:.2f}s || Estimated Time Remaining: {hr}h:{mr}m:{sr:.2f}s\n")
                generate(model, tokenizer)

                # Mini Batch Validation Pass Every 10 Print Intervals:
                model.eval()
                with torch.no_grad():
                    v_inputs, v_targets = next(iter(val_loader))
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    v_logits = model(v_inputs)
                    v_loss = criterion(v_logits.view(batch_size*context_len, vocab_size), v_targets.view(batch_size*context_len).long()).item()
                    val_losses.append(v_loss)
                    print(f"Mini-batch Validation Loss: {v_loss :.3f}\n")
                model.train()            
        
        # Full Validation Set Loss at the End:
        model.eval()
        print(f"Validating Loss Over {val_batches} Validation Batches...")
        acc = []
        with torch.no_grad():
            for v_inputs, v_targets in val_loader:
                v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                v_logits = model(v_inputs)
                v_loss += criterion(v_logits.view(batch_size*context_len, vocab_size), v_targets.view(batch_size*context_len).long()).item()
                acc.append(v_loss)
        epoch_loss = sum(acc) / len(acc)
        val_losses.append(epoch_loss)
        print("Validation Set Loss: {:.4f}".format(epoch_loss))
        model.train()
        save(cfg, model, optimizer, train_losses, val_losses, resume_from=0, e=epoch_number)

    elapsed = time.time() - start
    print(f"Training Complete")
    print(f"\nTotal Elapsed time: {elapsed}")
    print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
    print(f"Batch Size: {batch_size} || Learning Rate: {cfg.lr}")
    print(f"Context Length: {context_len}")
    print(f"Embedding Dimension: {model.embedding_dim}")
    print(f"Layers: {model.num_layers}")
    print(f"Heads per Layer: {model.heads_per_layer}\n")
    return model, optimizer #, training_log

def save(cfg, model, optimizer, train_losses=[], val_losses=[], resume_from=0, e=0):

    # TODO: This could use a little polishing
    filepath = os.path.join('__checkpoints', cfg.name, f'epoch_{resume_from+e+1}.net')
    os.makedirs(filepath, exist_ok=True)

    torch.save({
        'cfg': cfg,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
        }, filepath)

    # metadata
    path, filename = os.path.split(filepath)
    base, _ = os.path.splitext(filename)
    meta_filename = base + ".plot"
    metadata = {
        # Training Summary:
        'print_interval': cfg.print_interval,
        'validation_interval': cfg.validation_interval,
        #'output_samples': len(train_losses) + len(val_losses), # Idk about this
        'train_minibatch_losses': train_losses,#epoch_losses,
        'val_minibatch_losses': val_losses[:-1],
        'epoch_full_validation_batch_loss': val_losses[-1:],
    }
    os.makedirs(os.path.join(path, 'loss_plot'), exist_ok=True)
    with open(os.path.join(path, 'loss_plot', meta_filename), "w") as f:
        json.dump(metadata, f, indent=4)

def load(filepath, model, optimizer=None):
    # TODO: Make sure the model you're trying to load is compatible with the device you're on
    # TODO: Implement for CPU
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if not optimizer:
        optimizer=torch.optim.Adam(model.parameters(), lr=model.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return model, optimizer

def generate(model, tokenizer, prompt=[], max_length=500):
    # TESTING
    max_length = model.context_len

    # TODO: temperature, top-k (see ch 10)
    model.eval()
    tokens = torch.tensor([[tokenizer.start_token_idx] + prompt], device=model.device())
    for _ in range(max_length):
        logits = model(tokens)[:, -1:, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.distributions.Categorical(probs=probs).sample()
        tokens = torch.cat([tokens, next_token], dim=-1)
        if next_token == tokenizer.end_token_idx:
            break
    print(f"{tokenizer.decode(tokens[0].tolist())}\n")
    model.train()