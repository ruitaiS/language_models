import os
import data
from modules import LanguageModel

import time
import csv

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

base_path = os.path.dirname(os.path.abspath(__file__))

masked = True
batch_size = 16
seq_len = 8
input_batches, target_batches = data.get_sequences(batch_size, seq_len, dataset='train')

d = 8
num_layers = 6
total_heads = 2
#head_dim = embed_dim / total_heads
model = LanguageModel(d, data.get_vocab(), seq_len, num_layers, total_heads)
model.train()

print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
for p in model.parameters():
  p.requires_grad = True

#print(f"Parameters: {model.parameters()}")

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

print(f"Optimizer: {optimizer}")

total_batches = len(input_batches)
filename = f'model-02' # {int(time.time())}'
print_interval = 500

try:
	with open(os.path.join(base_path, f'models/{filename}_tr.txt'), 'w') as f:
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
	torch.save(model.state_dict(), os.path.join(base_path, f'models/{filename}.pth'))
f.close()
torch.save(model.state_dict(), os.path.join(base_path, f'models/{filename}.pth'))






#new_model = LanguageModel(d, vocab_size, seq_len, num_layers, total_heads)
#new_model.load_state_dict(torch.load(f"models/{filename}.pth"))
#new_model.train()

#for X, Y in zip(input_batches[-500:], target_batches[-500:]):
#	logits, loss = model(X, targets=Y)
#	print(f"Batch {batch_index} of {total_batches}. Loss: {loss}")
#	loss.backward()
#	optimizer.step()
#	batch_index += 1
