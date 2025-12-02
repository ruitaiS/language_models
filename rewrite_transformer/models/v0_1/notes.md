```
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
'lr': 5e-4 * (95 / 512), #  batch_size/512
'max_norm': 1.0,
'print_interval': 100,
'validation_interval': 100,
'weight_decay': 0.0,
'epochs': 5,
```

Trying the gpt2 clone. Same context length and still using character tokenization, but network dimensions otherwise in line with gpt2. batch size reduced until it wouldn't crash. also removed dropout (to check the issue with validation vs. train minibatch losses), and weight decay (because idk what it does yet)

7 hour epochs fml

have not addressed save / load / loss nuisances due to lizard brain