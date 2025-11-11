import os
import pandas as pd
import numpy as np
import Levenshtein
import math
import utils

vocab = utils.get_vocab()
bigram = utils.get_bigram()
transition_matrix = bigram.pivot(index='x_i', columns='x_j', values='normalized_P').fillna(0)
transition_matrix = transition_matrix.reindex(index=list(range(len(vocab))), columns=list(range(len(vocab))), method="pad", fill_value=0)
log_transition_matrix = np.log(transition_matrix.replace(0, np.finfo(float).tiny))

def word(word_index):
  if not isinstance(vocab.at[word_index, 'word'], str):
    print("not string")
    print(word_index)
  return vocab.at[word_index, 'word']

def logP_emission(observed, word_index, l = 0.01):
  #Use lowercase so that capitalization does not affect the emission probability
  #Ensures only transition probilities will determine whether to use the capitalized or lowercase form
  k = Levenshtein.distance(observed.lower(), word(word_index).lower())
  return k * math.log(l) - math.lgamma(k + 1) - l

def recurse(remaining, phrases, log_phrase_probs):
    log_phrase_probs = np.array(log_phrase_probs)
    observed = remaining.pop(0)
    print(f'Most Likely Preceeding Phrase: {phrases[np.argmax(log_phrase_probs)]}')
    print(f'Current Word: {observed}')
    print(f'Remaining Words: {remaining}')

    log_emission_probs = np.array([logP_emission(observed, word_index, l=0.01) for word_index in range(len(vocab))])

    #Multiply preceeding phrase probabilities *across* transition matrix rows (ith row scaled by ith phrase probability)
    #Multiply word emission probabilities *down* transition matrix columns (jth column scaled by jth emission probability)
    log_product_matrix = log_phrase_probs[:, None] + log_transition_matrix + log_emission_probs[None, :]


    # Columnwise idxmax and np.max gives the index and probability of most likely preceeding phrase for each word index
    # Discard all other phrases that end in the current word
    indices = np.argmax(log_product_matrix, axis=0).tolist()
    updated_phrases = [ phrases[phrase_index] + " " + word(word_index) for word_index, phrase_index in enumerate(indices)]
    updated_phrase_log_probs = np.max(log_product_matrix, axis=0)
    if remaining:
      return recurse(remaining, updated_phrases, updated_phrase_log_probs)
    else:
      print(f'Corrected Phrase: {updated_phrases[np.argmax(updated_phrase_log_probs)]}')
      return updated_phrases[np.argmax(updated_phrase_log_probs)]
   
#Sentences to Correct
E_1 = ['I','think','hat','twelve','thousand','pounds']

phrase_probs = pd.Series([0] * len(vocab))
phrase_probs[152] = 1

phrases = [""] * len(vocab)
phrases[152] = "<s>"

corrected_E1 = recurse(E_1, phrases, phrase_probs)