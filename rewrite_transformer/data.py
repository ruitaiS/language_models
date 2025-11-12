import os
import torch
import math
import pandas as pd
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer

base_path = os.path.dirname(os.path.abspath(__file__))

# TODO: Pretty sure i'm just going with char tokenization again b/c it's easier
def tokenize(text):
    tokenizer = RegexpTokenizer(r"<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
    words = tokenizer.tokenize(text)
    return words

def build_vocab():
    dataset_tokens = []
    for fileid in brown.fileids():
        text = brown.raw(fileid)
        tokens = tokenize(text)
        dataset_tokens.extend(tokens)
    vocab = sorted(set(token for token in dataset_tokens))

    vocab.insert(0, '<?>') # out of dictionary token
    vocab.insert(1, '<s>') # start token
    #vocab.insert(2, '</s>') # end token
    #vocab.insert(3, '<>') # pad token

    vocab_size = len(vocab)
    idx2token = dict(enumerate(vocab))
    token2idx = {word:i for i, word in idx2token.items()}
    return vocab_size, idx2token, token2idx

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

