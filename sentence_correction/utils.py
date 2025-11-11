import os
import pandas as pd
import numpy as np
import math
import Levenshtein
from nltk.tokenize import RegexpTokenizer

def logP_emission(observed, word, l = 0.01):
    k = Levenshtein.distance(observed.lower(), word.lower())
    return k * math.log(l) - math.lgamma(k + 1) - l

def recurse(remaining, phrases, log_phrase_probs, log_transition_matrix, idx2token):
    log_phrase_probs = np.array(log_phrase_probs) # TODO: Janky
    observed = remaining.pop(0)
    log_emission_probs = np.array([logP_emission(observed, word, l=0.01) for word in idx2token.values()])
    log_product_matrix = log_phrase_probs[:, None] + log_transition_matrix + log_emission_probs[None, :] # Explanation for this in ecse_526/p2.py line 35
    indices = np.argmax(log_product_matrix, axis=0).tolist()
    updated_phrases = [phrases[phrase_index] + [token_index] for token_index, phrase_index in enumerate(indices)]
    updated_phrase_log_probs = np.max(log_product_matrix, axis=0)

    # TODO: sometimes it outputs '<s>' and idk why
    token_ids = updated_phrases[np.argmax(updated_phrase_log_probs)]
    phrase = ' '.join([idx2token.get(token_id, '<?>') for token_id in token_ids])
    print(f"\r{phrase}", end="", flush=True)
    if not remaining: print("\n")

    if remaining:
        return recurse(remaining, updated_phrases, updated_phrase_log_probs, log_transition_matrix, idx2token)
    else:
        token_ids = updated_phrases[np.argmax(updated_phrase_log_probs)]
        return token_ids

def tokenize(text):
    tokenizer = RegexpTokenizer(r"<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
    words = tokenizer.tokenize(text)
    return words

def remove_verse_reference(dataset):
    output = []
    for line in dataset:
        parts = line.split("\t")
        if len(parts) > 1:
            text = parts[1].strip()
            output.append(text)
        else:
            print(f"Skipped line: {line}")
    return output

def extract_components():
    vocab = set()
    bigram_counts = {}
    bigram_totals = {}
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/akjv.txt')
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
        lines = remove_verse_reference(lines)
        for text in lines:
            words = ['<s>'] + tokenize(text) + ['</s>']
            vocab.update(words)
            for i in range(1, len(words)):
                bigram_counts[words[i-1], words[i]] = bigram_counts.get((words[i-1], words[i]), 0) + 1
                bigram_totals[words[i-1]] = bigram_totals.get(words[i-1], 0) + 1

    # Reorder and Add Special Tokens
    vocab.discard('<s>')
    vocab = sorted(vocab)
    vocab.insert(0, '<?>') # out of dictionary token
    vocab.insert(1, '<s>') # start token

    vocab_size = len(vocab)
    idx2token = dict(enumerate(vocab))
    token2idx = {word:i for i, word in idx2token.items()}

    bigram_probs = {bigram: count / bigram_totals[bigram[0]] for bigram, count in bigram_counts.items()}
    bigram_lp = {(token2idx[bigram[0]], token2idx[bigram[1]]) : math.log10(prob) for bigram, prob in bigram_probs.items()}
    bigram_lp = pd.DataFrame([{'x_i': x_i, 'x_j': x_j, 'e':e} for (x_i, x_j), e in bigram_lp.items()])

    # TODO: This becomes intractable for larger vocabularies (eg. Brown text corpus)
    # For bible text it's ok (10k vs. 50k vocab), but needs a pruning step for general use
    log_transition_matrix = bigram_lp.pivot(index = 'x_i', columns = 'x_j', values='e').fillna(float('-inf'))
    log_transition_matrix = log_transition_matrix.reindex(index=list(range(vocab_size)), columns=list(range(vocab_size)), method='pad', fill_value=float('-inf'))

    return vocab_size, idx2token, token2idx, log_transition_matrix