import os
import json
import torch

import data
from modules import LanguageModel
base_path = os.path.dirname(os.path.abspath(__file__))

def load_model(model_name = 'model-03'):

	filepath = os.path.join(base_path, f'models/{model_name}_meta.json') #os.join(base_path, 'models/{query}_meta.json')
	metadata = {}
	with open(filepath, 'r') as f:
		metadata = json.load(f)

	#xft, tfx = data.get_vocab(dataset_code = metadata['dataset_code'])
	model = LanguageModel(
		data.get_vocab(metadata['dataset_code']),
		metadata['d'],
		metadata['context_len'],
		metadata['num_layers'],
		metadata['total_heads'])
	model.load_state_dict(torch.load(os.path.join(base_path, f'models/{model_name}.pth')))
	model.eval()

	print(f"Model: {model_name}.pth")
	print(f"Context Length: {metadata['context_len']}, Layers: {metadata['num_layers']}, Heads: {metadata['total_heads']}")
	print(f"Vocab Size: {metadata['vocab_size']}")

	return model