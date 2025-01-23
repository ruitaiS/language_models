import pandas as pd
import numpy as np
import data

vocab = data.get_vocab()
unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()

def word(index):
  return vocab.at[index, 'word']

def unigram_next():
  return np.random.choice(unigram, size=1, p=unigram['normalized_P'])[0]

def bigram_next(x_i):
  subset = bigram[(bigram['x_i'] == x_i)]
  if subset.empty: raise ValueError(f"No bigrams begin with '{word(x_i)}'")
  return np.random.choice(subset['x_j'], size=1, p=subset['normalized_P'])[0]

def trigram_next(x_i, x_j):
  subset = trigram[(trigram['x_j'] == x_j) & (trigram['x_i'] == x_i)]
  if subset.empty:
    print("Trigram Error")
    raise ValueError(f"No trigrams begin with '{word(x_i)} {word(x_j)}'")
  return np.random.choice(subset['x_k'], size=1, p=subset['normalized_P'])[0]

# TODO: 
# This needs a rewrite so it actually traverses the search space intelligently

# rn it's also stuck on the edge case of <s> <x_0> not having any trigrams
# but won't back off to just <s> and pick another <x_0>

# consider weighted merging the bigram vs. trigram probs
# i think rn it looks exclusively at trigram probs

# Model Smoothing
# 10x all the counts, then add scalar 1
# Should keep proportions but get rid of absolute zero probability options
# Workaround for the generation function needing to back out


def recurse(word_indices):
  if (len(word_indices) == 0):
    x_0 = unigram_next()
    word_indices.append(x_0)
    return recurse(word_indices)
  elif (len(word_indices) == 1):
    try:
      x_i = word_indices[0]
      x_j = bigram_next(x_i)

      word_indices.append(x_j)
      #return word_indices
      return recurse(word_indices)
    except Exception as e:
      print(f"{e}\nBacking off to empty list")
      return recurse([])      
  else:
    try:
      x_i = word_indices[-2]
      x_j = word_indices[-1]
      x_k = trigram_next(x_i, x_j)
      word_indices.append(x_k)
    except Exception as e:
      print(f"{e}\nBacking off to: {word_indices[:-1]}")
      return recurse(word_indices[:-1])
    
  if word_indices[-1] == 151:
    return word_indices
  else:
    return recurse(word_indices)

#TODO: Generalize
def valid_sentence(word_indices):
  #Reject sentences that end without a valid character
  valid_endings = [1,4,5,6,7]
  return (word_indices[-2] in valid_endings)
  
#TODO: Generalize
def generate_sentence():
  #x_0 is first word, and is always 152: '<s>'
  #x_i is ith word, x_j is i+1 th word, x_k is i+2 th word
  #x_n is last word, and is always 151: '</s>'
  x_0 = 9 # TODO: Generalize. Should be able to read this value right off the vocab df, instead of hardcoding
  word_indices = recurse([x_0])
  if not valid_sentence(word_indices): return generate_sentence()
  sentence = ' '.join(map(lambda index: word(index), word_indices))
  print(sentence)
  return sentence
  
number_of_sentences = 10
sentences = [generate_sentence() for _ in range(0, number_of_sentences)]

with open('p1_output.txt', "w") as file:
  for sentence in sentences:
    file.write(f'{sentence}\n')