import os
from utils import Config, init, load, train
# Training Logic is in utils.py

simple = {
    # Transformer Parameters:
    'context_len': 32, # GPT2: 1024
    'embedding_dim': 32,
    'num_layers': 4,
    'heads_per_layer': 4,
    'ffn_expansion_ratio': 4, # 768 * 4 = 3072 FFN Hidden Dim

    # GPT2 uses no dropout btw!
    "embedding_dropout":0.1,
    "post_mha_dropout": 0.1,
    "post_ffn_dropout": 0.1,
    "attention_head_dropout": 0.1,

    # Data Parameters:
    'tk_method': 'char', # GPT2: BPE
    'include_book': True,

    'batch_size': 16, # GPT2: 512
    'validation_p': 0.1,
    'shuffle': True,
    'drop_last': True,
    'num_workers': 4, # or 8
    'pin_memory': False,
    'prefetch_factor': 2, # to 4
    'persistent_workers': True,

    # Training Parameters:
    'lr': 5e-4 * (16 / 512), #  batch_size/512
    'max_norm': 1.0,
    'print_interval': 100,
    'validation_interval': 100,
    'weight_decay': 0.1,
    'epochs': 2,
}

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

tweak = {
    'name': 'tweak',
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


gpu=True
if gpu:
    cfg = Config(tweak)
    #cfg = Config(default)
    cfg.device = 'cuda'
else:
    # This is a completely different architecture and training regimen
    # Will need to make sure cpu models and gpu models are kept apart
    cfg = Config(simple)
    cfg.device = 'cpu'

tokenizer, model, optimizer, criterion, train_loader, val_loader = init(cfg)

# TODO: Load specific epoch
#filepath = os.path.join('__checkpoints', cfg.name, f'epoch_15_0.net')
#model, optimizer = load(filepath, model, optimizer)

model, optimizer = train(cfg, tokenizer, model, optimizer, criterion, train_loader, val_loader)