import os
import torch
from modules import LanguageModel
import data

base_path = os.path.dirname(os.path.abspath(__file__))



def continuation(user_prompt):

	context_len = 8
	xft, tfx = data.get_vocab()

	d = 8
	vocab_size = len(xft)
	num_layers = 6
	total_heads = 2

	filename = 'model-02'
	new_model = LanguageModel(data.get_vocab(), d, context_len, num_layers, total_heads)
	new_model.load_state_dict(torch.load(os.path.join(base_path, f'models/{filename}.pth')))
	new_model.eval()

	print(f"Model: {filename}.pth")
	print(f"Context Length: {context_len}, Layers: {num_layers}, Heads: {total_heads}")
	print(f"Vocab Size: {vocab_size}")

	#user_prompt = [xft['<s>']]
	output = new_model.generate(user_prompt, response_length = 100)
	print(' '.join(output))

