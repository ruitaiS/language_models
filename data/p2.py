import pandas as pd
import numpy as np
import Levenshtein
import math
import data

vocab = data.get_vocab()
bigram = data.get_bigram()
transition_matrix = bigram.pivot(index='x_i', columns='x_j', values='normalized_P').fillna(0)
transition_matrix = transition_matrix.reindex(index=list(range(len(vocab))), columns=list(range(len(vocab))), method="pad", fill_value=0)

def word(word_index):
  return vocab.at[word_index, 'word']

def P_emission(observed, word_index, l = 0.01):
  #Use lowercase so that capitalization does not affect the emission probability
  #Ensures only transition probilities will determine whether to use the capitalized or lowercase form
  k = Levenshtein.distance(observed.lower(), word(word_index).lower())
  return ((l**k)*(math.e**(-l)))/math.factorial(k)

def recurse(remaining, phrases, phrase_probs):
    observed = remaining.pop(0)
    print(f'Most Likely Preceeding Phrase: {phrases[phrase_probs.idxmax()]}')
    print(f'Current Word: {observed}')
    print(f'Remaining Words: {remaining}')

    emission_probs = [P_emission(observed, word_index, l = 0.01) for word_index in range(len(vocab))]

    #Multiply preceeding phrase probabilities *across* transition matrix rows (ith row scaled by ith phrase probability)
    #Multiply word emission probabilities *down* transition matrix columns (jth column scaled by jth emission probability)
    product_matrix = np.diag(phrase_probs) @ transition_matrix @ np.diag(emission_probs)

    # Columnwise idxmax and np.max gives the index and probability of most likely preceeding phrase for each word index
    # Discard all other phrases that end in the current word
    indices = product_matrix.idxmax(axis=0).tolist()
    updated_phrases = [ phrases[phrase_index] + " " + word(word_index) for word_index, phrase_index in enumerate(indices)]
    updated_phrase_probs = np.max(product_matrix, axis=0)

    if remaining:
      return recurse(remaining, updated_phrases, updated_phrase_probs)
    else:
      print(f'Corrected Phrase: {updated_phrases[updated_phrase_probs.idxmax()]}')
      return updated_phrases[updated_phrase_probs.idxmax()]
   
#Sentences to Correct
E_1 = ['I','think','hat','twelve','thousand','pounds']
E_2 = ['she', 'haf', 'heard', 'them']
E_3 = ['She','was','ulreedy','quit','live']
E_4 = ['John', 'Knightly','wasn’t','hard','at','work']
E_5 = ['he','said','nit','word','by']

phrase_probs = pd.Series([0] * len(vocab))
phrase_probs[152] = 1

phrases = [""] * len(vocab)
phrases[152] = "<s>"

corrected_E1 = recurse(E_1, phrases, phrase_probs)
#print(corrected_E1)

corrected_E2 = recurse(E_2, phrases, phrase_probs)
#print(corrected_E2)

corrected_E3 = recurse(E_3, phrases, phrase_probs)
#print(corrected_E3)

corrected_E4 = recurse(E_4, phrases, phrase_probs)
#print(corrected_E4)

corrected_E5 = recurse(E_5, phrases, phrase_probs)
#print(corrected_E5)

print(corrected_E1)
print(corrected_E2)
print(corrected_E3)
print(corrected_E4)
print(corrected_E5)

'''
1.
P * E elementwise multiplication
diagonalize
multiply by transition matrix

get list of max indexes

each column represents a word
append the word to the phrase at the max index



'''