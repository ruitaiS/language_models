import os
import json
import data
from modules import LanguageModel

import time
import csv

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

base_path = os.path.dirname(os.path.abspath(__file__))

metadata = {
	'model_name': 'model-03',
	'dataset_code': 1741066844,
	'masked': True,
	'batch_size': 16,
	'context_len': 8,
	'd': 8,
	'num_layers': 6,
	'total_heads': 2,
}



model = LanguageModel(data.get_vocab(metadata['dataset_code']), metadata['d'], metadata['context_len'], metadata['num_layers'], metadata['total_heads'])
model.train()

print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
#print(f"Parameters: {model.parameters()}")
for p in model.parameters():
  p.requires_grad = True

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
print(f"Optimizer: {optimizer}")

input_batches, target_batches = data.get_sequences(metadata['batch_size'], metadata['context_len'], dataset='train')
total_batches = len(input_batches)
print_interval = 500

try:
	with open(os.path.join(base_path, f'models/{metadata['model_name']}_tr.txt'), 'w') as f:
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
				minutes = int(seconds / 60)
				seconds = int(seconds % 60)
				print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}. Estimated time remaining: {minutes}m{seconds}s")
except KeyboardInterrupt:
	f.close()
	torch.save(model.state_dict(), os.path.join(base_path, f'models/{metadata['model_name']}.pth'))
	with open(os.path.join(base_path, f'models/{metadata['model_name']}_meta.json'), 'w') as f:
		json.dump(metadata, f, indent=4)

f.close()
torch.save(model.state_dict(), os.path.join(base_path, f'models/{metadata['model_name']}.pth'))
with open(os.path.join(base_path, f'models/{metadata['model_name']}_meta.json'), 'w') as f:
	json.dump(metadata, f, indent=4)






#new_model = LanguageModel(vocab, d, context_len, num_layers, total_heads)
#new_model.load_state_dict(torch.load(f"models/{metadata.model_name}.pth"))
#new_model.train()

#for X, Y in zip(input_batches[-500:], target_batches[-500:]):
#	logits, loss = model(X, targets=Y)
#	print(f"Batch {batch_index} of {total_batches}. Loss: {loss}")
#	loss.backward()
#	optimizer.step()
#	batch_index += 1
