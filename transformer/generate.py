import os
import torch
from modules import LanguageModel
import data

base_path = os.path.dirname(os.path.abspath(__file__))

context_len = 8
xft, tfx = data.get_vocab()

d = 8
vocab_size = len(xft)
num_layers = 6
total_heads = 2

filename = 'model-0'
new_model = LanguageModel(d, data.get_vocab(), context_len, num_layers, total_heads)
new_model.load_state_dict(torch.load(os.path.join(base_path, f'models/{filename}.pth')))
new_model.eval()

print(f"Model: {filename}.pth")
print(f"Vocab Size: {vocab_size}")
print(f"Context Length: {context_len}")

output = new_model.generate(max_tokens = 100)
print(' '.join(output[7:]))