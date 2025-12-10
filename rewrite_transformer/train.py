import os
from utils import Config, init, load_model, train
# Training Logic is in utils.py

fast = {
    'name': 'fast',

    # Transformer Parameters:
    'context_len': 32, # GPT2: 1024
    'embedding_dim': 32,
    'num_layers': 4,
    'heads_per_layer': 4,
    'ffn_expansion_ratio': 4, # 768 * 4 = 3072 FFN Hidden Dim

    # GPT2 uses no dropout btw!
    "embedding_dropout":0.0, #0.1,
    "post_mha_dropout": 0.0, #0.1,
    "post_ffn_dropout": 0.0, #0.1,
    "attention_head_dropout": 0.0, #0.1,

    # Data Parameters:
    'tk_method': 'char', # GPT2: BPE
    'include_book': True,

    'batch_size': 30000, # cuda: 30000, cpu: ?
    'validation_p': 0.1,
    'shuffle': True,
    'drop_last': True,
    'num_workers': 4, # or 8
    'pin_memory': False,
    'prefetch_factor': 2, # to 4
    'persistent_workers': True,

    # Training Parameters:
    'lr': 5e-4 * (30000 / 512), #  batch_size/512
    'max_norm': 1.0,
    'print_interval': 10,
    'validation_interval': 10,
    'weight_decay': 0.1,
    'epochs': 50, # cuda: 1:15m per epoch # cpu: ?

    # Training state:
    'epoch': 0,
    'batch': 0,
}
# TODO:
'''
- loader state (tracking the batch number alone isn't enough)

- should seperate the values that will remain static the entire duration of the training,
vs. values that need to update during iteration loops (eg. epoch and batch number)
'''

simple_gpu = {
    'name': 'simple_gpu',

    # Transformer Parameters:
    'context_len': 192, # GPT2: 1024
    'embedding_dim': 128,
    'num_layers': 24,
    'heads_per_layer': 4,
    'ffn_expansion_ratio': 4, # expansion * embedding = FFN Hidden Dim

    # GPT2 uses no dropout btw!
    "embedding_dropout":0.1,
    "post_mha_dropout": 0.1,
    "post_ffn_dropout": 0.1,
    "attention_head_dropout": 0.1,

    # Data Parameters:
    'tk_method': 'char', # GPT2: BPE
    'include_book': True,

    'batch_size': 192, # GPT2: 512
    'validation_p': 0.1,
    'shuffle': True,
    'drop_last': True,
    'num_workers': 4, # or 8
    'pin_memory': True,
    'prefetch_factor': 2, # to 4
    'persistent_workers': True,

    # Training Parameters:
    'lr': 5e-4 * (192 / 512), #  batch_size/512
    'max_norm': 1.0,
    'print_interval': 100,
    'validation_interval': 100,
    'weight_decay': 0.1,
    'epochs': 10,
}

default = {
    'name': 'default',
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

gpt2like = {
    'name': 'gpt2-like',
    # Transformer Parameters:
    'context_len': 192, # GPT2: 1024
    'embedding_dim': 768,
    'num_layers': 12,
    'heads_per_layer': 12,
    'ffn_expansion_ratio': 4, # 768 * 4 = 3072 FFN Hidden Dim

    # GPT2 uses no dropout btw!
    "embedding_dropout":0,
    "post_mha_dropout": 0,
    "post_ffn_dropout": 0,
    "attention_head_dropout": 0,

    # Data Parameters:
    'tk_method': 'char', # GPT2: BPE
    'include_book': True,

    'batch_size': 95, # GPT2: 512
    'validation_p': 0.1,
    'shuffle': True,
    'drop_last': True,
    'num_workers': 4, # or 8
    'pin_memory': True,
    'prefetch_factor': 2, # to 4
    'persistent_workers': True,

    # Training Parameters:
    'lr': 5e-4 * (95 / 512), #  gpt2_lr * (batch_size / gpt2 batch_size)
    'max_norm': 1.0,
    'print_interval': 100,
    'validation_interval': 100,
    'weight_decay': 0.0,
    'epochs': 5,
}


resume=True
gpu=True
if gpu:
    cfg = Config(fast)
    cfg.device = 'cuda'
else:
    cfg = Config(fast)
    cfg.device = 'cpu'
print(f"Device: {cfg.device}\n")
print(f"Printing Updates Every {cfg.print_interval} batches || {cfg.print_interval * cfg.batch_size } sequences\n")

tokenizer, model, optimizer, criterion, train_loader, val_loader = init(cfg)
save_dir = os.path.join('__checkpoints', cfg.device, cfg.name)
if os.path.isdir(save_dir):
    if resume == True:
        cfg, model, optimizer = load_model(save_dir, model, optimizer)
        print(f"Successfully Resumed {cfg.device}/{cfg.name} from epoch {cfg.epoch} batch {cfg.batch}")
    else:
        print("\nResume set to false but previous saved models exist")
        print("Manually remove existing save folder, or set resume to True")
        exit()
model, optimizer = train(cfg, tokenizer, model, optimizer, criterion, train_loader, val_loader)


'''
current logic works but is confusing

rn initialization and loading are seperate, and train contains logic to differentiate between the two.
loading should instead be daisychained directly onto initialization

we should either have a save_dir specified, or not:
- if it's not specified, then load one of the presets
- if it is specified, then load a cfg, and as part of the init process, load the model
'''
