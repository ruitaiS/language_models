import torch
from modules import LanguageModel
import data


context_len = 8
xft, tfx = data.get_vocab()

d = 8
vocab_size = len(xft)
num_layers = 6
total_heads = 2

filename = 'model-1740434770'
new_model = LanguageModel(d, data.get_vocab(), context_len, num_layers, total_heads)
new_model.load_state_dict(torch.load(f"models/{filename}.pth"))
new_model.eval()

print(f"Model: {filename}.pth")
print(f"Vocab Size: {vocab_size}")
print(f"Context Length: {context_len}")

output = new_model.generate(max_tokens = 100)
print(' '.join(output))