import torch
from modules import LanguageModel
import data


seq_len = 4
xft, tfx = data.get_vocab()

d = 8
vocab_size = len(xft)
num_layers = 6
total_heads = 2


filename = 'model1'
new_model = LanguageModel(d, data.get_vocab(), seq_len, num_layers, total_heads)
new_model.load_state_dict(torch.load(f"models/{filename}.pth"))

new_model.generate(max_tokens = 10)