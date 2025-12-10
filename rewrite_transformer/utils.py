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

    def to_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'config.json'), "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Saved config to '{os.path.join(save_dir, 'config.json')}'")
    
    @classmethod
    def load(cls, save_dir):
        try:
            with open(os.path.join(save_dir, 'config.json'), "r") as f:
                return cls(json.load(f))
        except Exception as e:
            raise RuntimeError("Failed to load config") from e

def init(cfg, verbose=True):
    print(f"Loaded Config: {cfg}\n")

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
        print(f"Total Model Parameters: {(sum(p.nelement() for p in model.parameters())):,}")
        print(f"Optimizer: {optimizer}")
        print(f"Loss Function: {criterion}")
        print(f"Learning Rate: {cfg.lr}")
    return tokenizer, model, optimizer, criterion, train_loader, val_loader

def train(cfg, tokenizer, model, optimizer, criterion, train_loader, val_loader):

    # Instantiating Training Loop Variables:
    device = torch.device(cfg.device)
    epoch = cfg.epoch
    batch = cfg.batch
    epochs = cfg.epochs
    training_batches = len(train_loader)
    val_batches = len(val_loader)
    batch_size = cfg.batch_size
    context_len = cfg.context_len
    vocab_size = cfg.tokenizer.vocab_size
    print_interval = cfg.print_interval
    max_norm = cfg.max_norm

    # Train Start
    model.to(device)
    model.train()
    start = time.time()
    for epoch_number in range(epoch, epochs):
        train_losses = []
        val_losses = []
        #grad_norms = []

        # TODO: Sync the batch order; this only sets the number of batches to loop
        for batch_number, (inputs, targets) in enumerate(train_loader, start=batch): # TODO: seems to iterate past the end :(
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            flattened_logits = logits.view(batch_size*context_len, vocab_size)
            flattened_targets = targets.view(batch_size*context_len).long()
            loss = criterion(flattened_logits, flattened_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Save Layer Gradients
            '''
            if batch_number != 0 and (batch_number % print_interval == 0 or batch_number == training_batches-1):
                layer_grads = []
                for block in model.transformer_layers:
                    total = 0.0
                    for p in block.parameters():
                        if p.grad is not None:
                            total += p.grad.norm(2).item() ** 2
                    layer_grads.append(total ** 0.5)
                print(f"Grad Norms: {layer_grads}")
                grad_norms.append(layer_grads)
            '''

            clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            if batch_number != 0 and (batch_number % print_interval == 0 or batch_number == training_batches-1):
                elapsed_time = time.time() - start
                time_per_epoch = (elapsed_time / (training_batches * epoch_number + batch_number)) *(training_batches)
                time_per_session = time_per_epoch * epochs
                time_left_in_epoch = (time_per_epoch * (epoch_number + 1)) - elapsed_time
                time_left_in_session = time_per_session - elapsed_time

                print(f"\nEpoch {epoch_number+1} / {epochs} || {batch_number} / {training_batches-1} || Training Loss: {loss :.3f}\n")
                print(f" {int(elapsed_time // 3600):02d}h:{int((elapsed_time % 3600)//60):02d}m:{int(elapsed_time % 60):02d}s Elapsed\n")
                print(f" {int(time_left_in_epoch // 3600):02d}h:{int((time_left_in_epoch % 3600)//60):02d}m:{int(time_left_in_epoch % 60):02d}s Est. Time Left in Epoch")
                print(f" {int(time_per_epoch // 3600):02d}h:{int((time_per_epoch % 3600)//60):02d}m:{int(time_per_epoch % 60):02d}s Est. Time Per Epoch\n")
                print(f" {int(time_left_in_session // 3600):02d}h:{int((time_left_in_session % 3600)//60):02d}m:{int(time_left_in_session % 60):02d}s Est. Time Left")
                print(f" {int(time_per_session // 3600):02d}h:{int((time_per_session % 3600)//60):02d}m:{int(time_per_session % 60):02d}s Est. Total Time\n")

                train_losses.append(loss.item())    
                generate(model, tokenizer)

                # Mini Batch Validation Pass
                model.eval()
                with torch.no_grad():
                    v_inputs, v_targets = next(iter(val_loader))
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    v_logits = model(v_inputs)
                    v_loss = criterion(v_logits.view(batch_size*context_len, vocab_size), v_targets.view(batch_size*context_len).long()).item()
                    val_losses.append(v_loss)
                    print(f"Mini-batch Validation Loss: {v_loss :.3f}")
                model.train()
                if batch_number != training_batches - 1:
                    cfg.batch = batch_number
                    save_model(cfg, model, optimizer)
        
        # Full Validation Set Loss at the End:
        model.eval()
        print(f"\nValidating Loss Over {val_batches} Validation Batches...")
        acc = []
        with torch.no_grad():
            for v_inputs, v_targets in val_loader:
                v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                v_logits = model(v_inputs)
                v_loss = criterion(v_logits.view(batch_size*context_len, vocab_size), v_targets.view(batch_size*context_len).long()).item()
                acc.append(v_loss)
        epoch_loss = sum(acc) / len(acc)
        val_losses.append(epoch_loss)
        print("Validation Set Loss: {:.4f}".format(epoch_loss))
        model.train()
        cfg.epoch, cfg.batch = epoch_number + 1, 0
        save_model(cfg, model, optimizer)

    elapsed = time.time() - start
    print(f"Training Complete")
    print(f"\nTotal Elapsed time: {int(elapsed // 3600):02d}h:{int((elapsed % 3600)//60):02d}m:{int(elapsed % 60):02d}s")
    print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
    print(f"Batch Size: {batch_size} || Learning Rate: {cfg.lr}")
    print(f"Context Length: {context_len}")
    print(f"Embedding Dimension: {model.embedding_dim}")
    print(f"Layers: {model.num_layers}")
    print(f"Heads per Layer: {model.heads_per_layer}\n")
    return model, optimizer #, training_log
#----------------------------

def save_model(cfg, model, optimizer):
    save_dir = os.path.join('__checkpoints', cfg.device, cfg.name)
    cfg.save(save_dir)

    if cfg.batch == 0:
        model_filename = f'epoch_{cfg.epoch}.net'
    else:
        model_filename = 'checkpoint.net'

    print(f"Saving '{os.path.join(save_dir, model_filename)}'\n")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
        }, os.path.join(save_dir, model_filename))
    
    if model_filename == 'checkpoint.net':
        # TODO: Save loader state so it can be resumed mid-epoch
        pass
    else:
        if os.path.isfile(os.path.join(save_dir, 'checkpoint.net')):
            os.remove(os.path.join(save_dir, 'checkpoint.net'))

# TODO: Loader state
def load_model(save_dir, model, optimizer, loader=None):
    # TODO: save and grab .cfg file
    cfg = Config.load(save_dir)

    if os.path.isfile(os.path.join(save_dir, 'checkpoint.net')):
        model_file = 'checkpoint.net' # TODO: Find + Update the loader state as well
        if not loader:
            # TODO: Warn that loader state isn't updated (will mess up loss function tracking)
            # Or maybe exit, idk
            pass
    else:
        model_file = f'epoch_{cfg.epoch}.net'
    filepath = os.path.join(save_dir, model_file)
    checkpoint = torch.load(filepath, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state'])
    if not optimizer:
        print(f"No optimizer recieved. Initializing Adam..")
        optimizer=torch.optim.Adam(model.parameters(), lr=model.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    # TODO: This shouldn't be necessary if you have it set up correctly
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(cfg.device)

    return cfg, model, optimizer
    
def update_plot(cfg, row):
    plot_filename = 'training_metadata.csv'
    '''
    - time
    - epoch
    - batch
    - training minibatch loss
    - validation minibatch loss
    - grad norms for each layer, one column for each layer
    - full pass validation loss (only once at the end of the epoch, else None)
    '''
    pass

def generate(model, tokenizer, prompt=[], max_length=500):
    # TESTING
    max_length = 200
    #max_length = model.context_len

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
