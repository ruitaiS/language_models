import torch
import math
from modules import LanguageModel
import data


context_len = 8
xft, tfx = data.get_vocab()

d = 8
vocab_size = len(xft)
num_layers = 6
total_heads = 2

masked = True
batch_size = 16
seq_len = 8

filename = 'model-1740437775'
model = LanguageModel(d, data.get_vocab(), context_len, num_layers, total_heads)
model.load_state_dict(torch.load(f"models/{filename}.pth"))
model.eval()

print(f"Model: {filename}.pth")
print(f"Vocab Size: {vocab_size}")
print(f"Context Length: {context_len}")

eval_input, eval_target = data.get_sequences(batch_size, seq_len, dataset='validation')
total_loss = 0.0
num_batches = 0

print(f"eval_input shape: {eval_input.shape}")
print(f"eval_target shape: {eval_target.shape}")
with torch.no_grad():
	for batch_index, (X, Y) in enumerate(zip(eval_input, eval_target)):
		logits, loss = model(X, targets=Y)
		print(f"Batch Index: {batch_index}, Loss: {loss.item()}")
		total_loss += loss.item()
		num_batches += 1

average_loss = total_loss / num_batches
perplexity = math.exp(average_loss)
print(f'Evaluation Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}')