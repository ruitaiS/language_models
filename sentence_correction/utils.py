import pandas as pd
import numpy as np
import math
import Levenshtein
from scipy.sparse import coo_matrix
import nltk
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
nltk.download('brown')

def logP_emission(observed, word, l = 0.01):
    k = Levenshtein.distance(observed.lower(), word.lower())
    return k * math.log(l) - math.lgamma(k + 1) - l

def recurse(remaining, phrases, log_phrase_probs, log_transition_matrix, idx2token):
    log_phrase_probs = np.array(log_phrase_probs) # TODO: check if you can remove this
    observed = remaining.pop(0)
    log_emission_probs = np.array([logP_emission(observed, word, l=0.01) for word in idx2token.values()])


    log_product_matrix = log_phrase_probs[:, None] + log_transition_matrix + log_emission_probs[None, :] # Explanation for this in ecse_526/p2.py line 35
    indices = np.argmax(log_product_matrix, axis=0).tolist()
    # TODO: Remove below after confirmed useless (feeds strings instead of token indexes as sequences)
    #updated_phrases = [phrases[phrase_index] + [tfx.get(token_index, '<?>')] for token_index, phrase_index in enumerate(indices)]
    updated_phrases = [phrases[phrase_index] + [token_index] for token_index, phrase_index in enumerate(indices)]
    updated_phrase_log_probs = np.max(log_product_matrix, axis=0)

    # TODO: sometimes it outputs '<s>' and idk why
    if idx2token is not None: # eg. should print
        token_ids = updated_phrases[np.argmax(updated_phrase_log_probs)]
        phrase = ' '.join([idx2token.get(token_id, '<?>') for token_id in token_ids])
        print(f"\r{phrase}", end="", flush=True)
        if not remaining: print("\n")

    if remaining:
        return recurse(remaining, updated_phrases, updated_phrase_log_probs, idx2token)
    else:
        token_ids = updated_phrases[np.argmax(updated_phrase_log_probs)]
        return token_ids

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

def extract_components():
    vocab = set()
    bigram_counts = {}
    bigram_totals = {}
    for sentence in brown.sents():
        # Recombine to string, and use our tokenizer instead
        text = " ".join(sentence)
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
    #vocab.insert(2, '</s>') # end token
    #vocab.insert(3, '<>') # pad token

    vocab_size = len(vocab)
    idx2token = dict(enumerate(vocab))
    token2idx = {word:i for i, word in idx2token.items()}

    bigram_probs = {bigram: count / bigram_totals[bigram[0]] for bigram, count in bigram_counts.items()}
    bigram_lp = {(token2idx[bigram[0]], token2idx[bigram[1]]) : math.log10(prob) for bigram, prob in bigram_probs.items()}
    bigram_lp = pd.DataFrame([{'x_i': x_i, 'x_j': x_j, 'e':e} for (x_i, x_j), e in bigram_lp.items()])

    log_transition_matrix = None # TODO (bigger vocab makes this a problem)
    '''
    rows = bigram_lp['x_i'].to_numpy()
    cols = bigram_lp['x_j'].to_numpy()
    vals = bigram_lp['e'].to_numpy()
    log_transition_matrix = coo_matrix((vals, (rows, cols)), shape=(vocab_size, vocab_size))
    '''

    return vocab_size, idx2token, token2idx, log_transition_matrix