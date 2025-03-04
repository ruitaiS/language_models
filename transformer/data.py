import os
import csv
import random
import torch
import math
import pandas as pd
from tokenizer import tokenize

base_path = os.path.dirname(os.path.abspath(__file__))

# TODO: everything calling on this should provide the dataset code
# TODO: Note this code will break if requesting any dataset other than training, due to 'a1'
# This, and the fact that this hasn't caused an issue, means that only the training set is ever requested
# so "dataset_name" is an irrelevant/unused parameter
def get_dataset(dataset_name, dataset_code):
	dataset = []
	with open(os.path.join(base_path, f'text/{dataset_code}/a1_{dataset_name}_set.txt'), 'r') as data_file:
		for line in data_file:
			tokens = tokenize(line)
			tokens = ["<s>"] + tokens + ["</s>"]
			dataset.append(tokens)
	return dataset

def get_bigram(dataset_code):
	dataset = get_dataset(dataset_name='train', dataset_code=dataset_code)
	xft, _ = get_vocab(dataset_code)
	bigram_counts = {}
	bigram_totals = {}
	for tokens in dataset:
		for i in range(1, len(tokens)):
			bigram_counts[tokens[i-1], tokens[i]] = bigram_counts.get((tokens[i-1], tokens[i]), 0) + 1
			bigram_totals[tokens[i-1]] = bigram_totals.get(tokens[i-1], 0) + 1

	bigram_probs = {bigram: count / bigram_totals[bigram[0]] for bigram, count in bigram_counts.items()}
	bigram_lp = {(xft[bigram[0]], xft[bigram[1]]) : math.log10(prob) for bigram, prob in bigram_probs.items()}
	'''
	with open(os.path.join(base_path, f'text/b1_bigram_counts.txt'), 'w') as f:
		writer = csv.writer(f, delimiter=' ')
		for (x_i, x_j), e in bigram_lp.items():
			writer.writerow([x_i, x_j, e])

	Bigram Length: 120685
	Bigram Dimensions: (120685, 4)
	Bigram Dimensions: Index(['x_i', 'x_j', 'normalized_P', 'e'], dtype='object')
	'''
	bigram_lp = pd.DataFrame([{'x_i': x_i, 'x_j': x_j, 'e':e} for (x_i, x_j), e in bigram_lp.items()])
	return bigram_lp


def get_vocab(dataset_code):
	dataset = get_dataset(dataset_name = 'train', dataset_code = dataset_code)
	vocab = set(['<s>', '</s>', '<?>', '<>'])
	for tokens in dataset:
		# print(f"Words in line: {len(line)}")
		vocab.update(tokens)
	# Assign index mapping and create vocab hash
	xft = {token : index for index, token in enumerate(sorted(vocab))} # index from token
	tfx = {index : token for index, token in enumerate(sorted(vocab))}# token from index

	print(f"{len(vocab)} words in training set vocab")

	with open(os.path.join(base_path, f'text/{dataset_code}/b0_vocab.txt'), 'w') as f:
		writer = csv.writer(f, delimiter=' ')
		writer.writerows(tfx.items()) # item = (index, token); sort by token

	return xft, tfx

def get_sequences(batch_size, context_len, dataset_code, shuffle=True, dataset='train'):
	'''
	context_len = 4

	sentence = [<s>, a, b, c, d, </s>] # length = 6
	token_id =               <s>,                  a,                 b,                c,              d,           </s>
	input =  [[<>, <>,  <>,  <>], [<>, <>,  <>, <s>], [<>,  <>, <s>, a], [ <>, <s>, a, b], [<s>, a, b, c], [a, b, c,    d]]
	target = [[<>, <>,  <>, <s>], [<>, <>, <s>,   a], [<>, <s>,   a, b], [<s>,   a, b, c], [  a, b, c, d], [b, c, d, </s>]]
  '''
	xft, _ = get_vocab(dataset_code)
	dataset = get_dataset(dataset, dataset_code)
	dataset = [[xft.get(token, xft["<?>"]) for token in sentence] for sentence in dataset]
	num_samples = sum(len(sentence) - 1 for sentence in dataset)

	inputs = [None]*num_samples
	targets = [None]*num_samples

	insert_idx = 0
	for sentence in dataset:
		sequence = [xft['<>']] * (context_len - 1 ) + [xft['<s>']]
		for token_id in sentence[1:]:
			inputs[insert_idx] = sequence.copy()
			next_sequence = sequence[1:] + [token_id]
			targets[insert_idx] = next_sequence
			#print(targets[insert_idx])
			sequence = next_sequence
			insert_idx += 1
	print(f"Sequences expected: {num_samples}, sequences created: {insert_idx}")

	# num_samples, context_len
	inputs = torch.tensor(inputs, dtype=torch.long)
	targets = torch.tensor(targets, dtype=torch.long)
	print(f"Input Shape: {inputs.shape}")
	print(f"Target Shape: {targets.shape}")

	if shuffle:
		reorder = torch.randperm(len(inputs))
		inputs = inputs[reorder]
		targets = targets[reorder]

	remainder = num_samples % batch_size
	if remainder:
		inputs = inputs[:-remainder]
		targets = targets[:-remainder]

	input_batches = inputs.view(-1, batch_size, context_len)
	target_batches = targets.view(-1, batch_size, context_len)

	print(f"Input Batches: {input_batches.shape}")
	print(f"Target Batches: {target_batches.shape}")

	return input_batches, target_batches

