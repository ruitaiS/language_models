import os
import json
import torch

import data
from modules import LanguageModel
base_path = os.path.dirname(os.path.abspath(__file__))

def load_model(model_name = 'model-03', mode='eval'):
	meta_fp = os.path.join(base_path, f'models/{model_name}_meta.json')
	model_fp = os.path.join(base_path, f'models/{model_name}.pth')
	metadata = {}
	try:
		with open(meta_fp, 'r') as f:
			metadata = json.load(f)
	except Exception as e:
		print(e)
		print(f'Metadata for model {model_name} not found')
		return None
	model = LanguageModel(
		data.get_vocab(metadata['dataset_code']),
		metadata['d'],
		metadata['context_len'],
		metadata['num_layers'],
		metadata['total_heads'])
	model.load_state_dict(torch.load(model_fp))

	modes = {
		'eval' : model.eval,
		'train': model.train,
	}
	assert mode in modes.keys(), f'Invalid operation mode: {mode}'
	modes[mode]()
	#model.eval()

	print(f"Model: {model_name}.pth")
	print(f"Context Length: {metadata['context_len']}, Layers: {metadata['num_layers']}, Heads: {metadata['total_heads']}")
	print(f"Operation Mode: {mode}")

	xft, tfx = data.get_vocab(dataset_code = metadata['dataset_code'])
	bigram_lp = data.get_bigram(dataset_code = metadata['dataset_code'])
	print(f"Vocab Size: {len(xft)}")

	return {
		'model': model,
		'vocab': (xft, tfx),
		'bigram_lp': bigram_lp
	}

test_model = load_model('model-01', 'eval')
print(f"Test model: {test_model}")
