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
        grad_norms = []
        for batch_number, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            flattened_logits = logits.view(batch_size*context_len, vocab_size)
            flattened_targets = targets.view(batch_size*context_len).long()
            loss = criterion(flattened_logits, flattened_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Save Layer Gradients
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

            clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            

            if batch_number != 0 and (batch_number % print_interval == 0 or batch_number == training_batches-1):
                elapsed = time.time() - start
                estimated = elapsed * (training_batches)/(batch_number)
                remaining = estimated * (epochs) - elapsed
                h, hr, eh = int(estimated // 3600), int(remaining // 3600), int(elapsed // 3600)
                m, mr, em = int((estimated % 3600) // 60), int((remaining % 3600) // 60), int((elapsed % 3600) // 60)
                s, sr, es = estimated % 60, remaining % 60, elapsed % 60

                train_losses.append(loss.item())    
                print(f"\n Epoch {epoch_number+1} / {epochs} || {batch_number} / {training_batches-1} || {eh}h:{em}m:{es:.2f}s || Loss: {loss :.3f}")
                print(f"Elapsed: {eh}h:{em}m:{es:.2f}s || Estimated Time Per Epoch: {h}h:{m}m:{s:.2f}s || Estimated Time Remaining: {hr}h:{mr}m:{sr:.2f}s\n")
                generate(model, tokenizer)

                # Mini Batch Validation Pass
                model.eval()
                with torch.no_grad():
                    v_inputs, v_targets = next(iter(val_loader))
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    v_logits = model(v_inputs)
                    v_loss = criterion(v_logits.view(batch_size*context_len, vocab_size), v_targets.view(batch_size*context_len).long()).item()
                    val_losses.append(v_loss)
                    print(f"Mini-batch Validation Loss: {v_loss :.3f}\n")
                model.train()
                save(cfg, model, optimizer, train_losses, val_losses, grad_norms, resume_from=0, e=epoch_number, b=batch_number)
        
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
        save(cfg, model, optimizer, train_losses, val_losses, grad_norms, resume_from=0, e=epoch_number)

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

def save(cfg, model, optimizer, train_losses=[], val_losses=[], grad_norms = [], resume_from=0, e=0, b=0):
    '''
    NOTE: Save/load process is completely fucked rn. Should be:

    Save:
    0) Create a subdirectory in checkpoints with the cfg and training statistics
        - __checkpoints/{cfg.name}
        - config.cfg
        - stats.dat, with columns:
            - time
            - epoch
            - batch
            - training minibatch loss
            - validation minibatch loss
            - grad norms for each layer, one column for each layer
            - full pass validation loss (only once at the end of the epoch, else None)
    1) Every print interval:
        - Append training statistics by line to a single file
        - Save checkpoint to epoch_x_batch_y_checkpoint.net
            - delete the last one when writing new one (or only keep every 5 or 10 or whatever. easiest to just delete)
    2) Every epoch:
        - save checkpoint to epoch_x.net
        - delete any intervening checkpoints or put them somewhere that doesn't clutter
        - calculate / write full pass loss

    Load:
    0) Load a model by name if specified, otherwise load the defaults
    1) Load CFG file first, so we know which device to send to
    2) Loader should call init
        - rn init is called first, and the created resources are passed to the loader for it to update
        - really this should be handled within loader, rather than exposed
    3) Trainer should know where the checkpoint left off
        - i don't want to handle resumes of batches between epochs, so let's just be ok with repeating training batches
        - but the filenames should not conflict
    '''

    #path = os.path.join('__checkpoints', cfg.name)
    #name = f'epoch_{resume_from+e+1}_{b}'




    # TODO: This could use a little polishing
    filepath = os.path.join('__checkpoints', cfg.name, f'epoch_{resume_from+e+1}_{b}.net')
    path, filename = os.path.split(filepath)
    base, _ = os.path.splitext(filename)
    meta_filename = base + ".plot"
    os.makedirs(path, exist_ok=True)



    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
        }, filepath)

    # metadata
    # TODO: clean
    epoch_loss = 0
    if b == 0:
        epoch_loss = val_losses[-1:]
        val_losses = val_losses[:-1]
    
    metadata = {
        # Training Summary:
        'print_interval': cfg.print_interval,
        'validation_interval': cfg.validation_interval,
        #'output_samples': len(train_losses) + len(val_losses), # Idk about this
        'train_minibatch_losses': train_losses,#epoch_losses,
        'val_minibatch_losses': val_losses,
        'gradient_norms': grad_norms,
        'epoch_full_validation_batch_loss': epoch_loss,
    }
    os.makedirs(os.path.join(path, 'loss_plot'), exist_ok=True)
    with open(os.path.join(path, 'loss_plot', meta_filename), "w") as f:
        json.dump(metadata, f, indent=4)

def load(filepath, model, optimizer=None):
    # TODO: This is lowk really janky
    # checkpoint contains the config, but config also contains the device that everything needs to go on
    # 
    checkpoint = torch.load(filepath, map_location='cuda')
    model.load_state_dict(checkpoint['model_state'])
    if not optimizer:
        print(f"No optimizer recieved. Initializing Adam..")
        optimizer=torch.optim.Adam(model.parameters(), lr=model.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    # TODO: This shouldn't be necessary if you have it set up correctly
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to('cuda')

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