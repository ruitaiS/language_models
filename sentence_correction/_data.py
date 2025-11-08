import os
import csv
import random
import torch
import math
import pandas as pd
import numpy as np
from tokenizer import tokenize

base_path = os.path.dirname(os.path.abspath(__file__))

# TODO: everything calling on this should provide the dataset code
# TODO: Note this code will break if requesting any dataset other than training, due to 'a1'
# This, and the fact that this hasn't caused an issue, means that only the training set is ever requested
# so "split_name" is an irrelevant/unused parameter
def get_dataset(split_name, time_code):
    dataset = []
    with open(os.path.join(base_path, f'text/{time_code}/{split_name}_set.txt'), 'r') as data_file:
        for line in data_file:
            tokens = tokenize(line)
            tokens = ["<s>"] + tokens + ["</s>"]
            dataset.append(tokens)
    return dataset

def extract_aux(dataset):
    # Happens before being passed in:
    # dataset = get_dataset(split_name = 'train', time_code = time_code)
    vocab = set(['<s>', '</s>', '<?>', '<>'])
    bigram_counts = {}
    bigram_totals = {}
    for tokens in dataset:
        vocab.update(tokens)
        for i in range(1, len(tokens)):
            bigram_counts[tokens[i-1], tokens[i]] = bigram_counts.get((tokens[i-1], tokens[i]), 0) + 1
            bigram_totals[tokens[i-1]] = bigram_totals.get(tokens[i-1], 0) + 1

    # print(f"{len(vocab)} words in training set vocab")
    xft = {token : index for index, token in enumerate(sorted(vocab))} # index from token
    tfx = {index : token for index, token in enumerate(sorted(vocab))}# token from index

    bigram_probs = {bigram: count / bigram_totals[bigram[0]] for bigram, count in bigram_counts.items()}
    bigram_lp = {(xft[bigram[0]], xft[bigram[1]]) : math.log10(prob) for bigram, prob in bigram_probs.items()}
    bigram_lp = pd.DataFrame([{'x_i': x_i, 'x_j': x_j, 'e':e} for (x_i, x_j), e in bigram_lp.items()])

    return {
            'vocab': (xft, tfx),
            'bigram': bigram_lp
            }

def batch(batch_size, context_len, dataset, shuffle=True):
    xft, _ = extract_aux(dataset)['vocab']
    dataset = [[xft.get(token, xft["<?>"]) for token in sentence] for sentence in dataset]

    num_samples = sum(len(sentence) - 1 for sentence in dataset)

    inputs = [None]*num_samples
    targets = [None]*num_samples

    insert_idx = 0
    for sentence in dataset:
        sequence = [xft['<>']] * (context_len - 1 ) + [xft['<s>']] 
        for token in sentence[1:]:
            inputs[insert_idx] = sequence.copy()
            next_sequence = sequence[1:] + [token]
            targets[insert_idx] = next_sequence
            sequence = next_sequence
            insert_idx += 1
    print(f"Sequences expected: {num_samples}, sequences created: {insert_idx}")

    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    #inputs = np.array(inputs)
    #targets = np.array(targets)
    # num_samples, context_len
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

    #input_batches = inputs.reshape(-1, batch_size, context_len)
    #target_batches = targets.reshape(-1, batch_size, context_len)
    input_batches = inputs.view(-1, batch_size, context_len)
    target_batches = targets.view(-1, batch_size, context_len)

    print(f"Input Batches: {input_batches.shape}")
    print(f"Target Batches: {target_batches.shape}")

    return input_batches, target_batches

