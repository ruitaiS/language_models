import pandas as pd
import numpy as np
import data
import math
from scipy.optimize import minimize

#vocab = data.get_vocab()
unigram = data.get_unigram() # (index) : log prob
bigram = data.get_bigram() # (index, index) : log prob
trigram = data.get_trigram() # (index, index, index) : log prob
xft, tfx = data.get_lookups() # index from token, token from index
observed_set = data.get_dev_set()

def get_index(t_i):
  if t_i in xft:
    return xft[t_i]
  else:
    # TODO: Decide on how to handle upper / lowercase versions
    # I think it should fall back to lowercase if it can't find the uppercase
    # (?) store probabilities for both cases, and compute combined when needed
    #print(f"Not in vocab: {t_i}")
    return -1

# Loss Function:
def negative_log_likelihood(params): # equiv to 1e-1000 raw prob
  l1, l2, l3, fallback = params
  print(f"l1: {params[0]}, l2: {params[1]}, l3: {params[2]}, fallback: {params[3]}")
  total_likelihood = 0

  for sentence in observed_set:
    sentence_likelihood = 0
    # t_i for token string representation, x_i for vocab index representation
    for i, t_i in enumerate(sentence):
      x_i = get_index(t_i)
      if x_i == -1:
        #print(f"Word {t_i} not in vocab")
        unigram_prob = fallback # log prob
        bigram_prob = fallback #
        trigram_prob = fallback #
      else:
        unigram_prob = unigram.get(x_i, fallback) # log prob
        # Note: <s> and </s> are removed from unigram prob
        # fallback value is needed even though these tokens exist in vocab
        #print(f"unigram: {unigram.get(x_i, fallback)}, fallback: {fallback}")
        if i == 1: # First word in the sentence after <s>
          bigram_prob = bigram.get((xft['<s>'], x_i), fallback)
          trigram_prob = fallback
        else:
          x_i_min_1 = get_index(sentence[i-1])
          bigram_prob = bigram.get((x_i_min_1, x_i), fallback) # log prob
          x_i_min_2 = sentence[i-2]
          trigram_prob = trigram.get((x_i_min_2, x_i_min_1, x_i), fallback)
      #print(f"Uni: {unigram_prob}, Bi: {bigram_prob}, Tri: {trigram_prob}")
      raw_prob = l1 * (10**unigram_prob) + l2 * (10**bigram_prob) + l3 * (10**trigram_prob)
      #print(f"l1: {l1}, l2: {l2}, l3: {l3}")
      #print(f"Raw: {raw_prob}, fallback: {1e-10}, max: {max(raw_prob, 1e-10)}")
      log_prob = math.log10(max(raw_prob, 1e-10))
      sentence_likelihood += log_prob
    total_likelihood += sentence_likelihood
  print(total_likelihood/len(observed_set))
  return -(total_likelihood/len(observed_set))


# Optimization
initial_params = [1/3, 1/3, 1/3, -100]
constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:3]) - 1}]  # l1 + l2 + l3 = 1
bounds = [(0, 1), (0, 1), (0, 1), (-1000, 0)]  # Each lambda must be between 0 and 1, fallback can go to -1000

# Minimize the negative log-likelihood
result = minimize(negative_log_likelihood, initial_params, 
                  constraints=constraints, bounds=bounds)

print(f"Success? : {result.success}")
print(f"Message : {result.message}")

# Optimal weights
optimal_lambdas = result.x
print("Optimal lambdas:", optimal_lambdas)

# [9.49650121e-01 9.37157253e-18 5.03498791e-02]
#l1: 0.19294876040543507, l2: 0.8070512544957388, l3: 0.0

#testval = negative_log_likelihood(initial_lambdas)
#print(testval)