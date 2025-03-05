import os
import time
import csv
import json
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

import datanew as data
from modules import LanguageModel
base_path = os.path.dirname(os.path.abspath(__file__))

def train_model(model_name,
                dataset_id,
                batch_size, # might be irrelevant for the model
                context_len,
                embedding_depth,
                num_layers,
                total_heads,
                masked = True):

    dataset = data.get_dataset('train', dataset_id)
    aux_data = data.extract_aux(dataset)

    model = LanguageModel(vocab = (aux_data['xft'], aux_data['tfx']),
                          d = embedding_depth, context_len = context_len, num_layers=num_layers, total_heads=total_heads)
    model.train()

    print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    print(f"Optimizer: {optimizer}")

    input_batches, target_batches = data.batch(batch_size, context_len, dataset)
    total_batches = len(input_batches)
    print_interval = 500

    try:
        with open(os.path.join(base_path, f'models/{model_name}_tr.txt'), 'w') as f:
            writer = csv.writer(f, delimiter=' ')

            start = time.time()
            for batch_index, (X, Y) in enumerate(zip(input_batches, target_batches)):
                logits, loss = model(X, targets=Y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                elapsed = time.time() - start
                writer.writerow([batch_index, loss.item(), elapsed])
                if batch_index % print_interval == 1:
                    seconds = (elapsed / batch_index)*(total_batches - batch_index)
                    minutes = int(seconds)
                    seconds = int(seconds % 60)
                    print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}. Estimated time remaining: {minutes}m{seconds}s")
    except KeyboardInterrupt:
        f.close()
        torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
        with open(os.path.join(base_path, f"models/{model_name}_meta.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

    f.close()
    torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
    with open(os.path.join(base_path, f"models/{model_name}_meta.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    return model

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
		data.get_vocab(metadata['dataset_id']),
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

	xft, tfx = data.get_vocab(dataset_id = metadata['dataset_id'])
	bigram_lp = data.get_bigram(dataset_id = metadata['dataset_id'])
	print(f"Vocab Size: {len(xft)}")

	return {
		'core': model,
		'vocab': (xft, tfx),
		'bigram_lp': bigram_lp
	}
