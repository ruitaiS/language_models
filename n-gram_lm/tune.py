import pandas as pd
import numpy as np
import data
import math
from scipy.optimize import minimize

#vocab = data.get_vocab()
unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
xft, tfx = data.get_lookups() # index from token, token from index
observed_set = data.get_dev_set()

def get_index(t_i):
  if t_i in xft:
    return xft[t_i]
  else:
    # TODO: Decide on how to handle upper / lowercase versions
    #print(f"Not in vocab: {t_i}")
    return -1

# Loss Function:
def negative_log_likelihood(lambdas, epsi = 1e-10):
  l1, l2, l3 = lambdas
  print(f"l1: {l1}, l2: {l2}, l3: {l3}")
  total_likelihood = 0

  # TODO: I think epsi gets doubled (if there's no bigram, there's obviously no trigram)




  for sentence in observed_set:
    for i, t_i in enumerate(sentence): # Let's adopt convention: t_i for token string representation, x_i for vocab index representation
      unigram_prob = unigram.get(get_index(t_i), -100)

      if i == 0:
          # TODO I think this is correct but revisit this
          # I think about this as dropping out the bigram / trigram terms completely
          # It messes up the comparisons, but not for the final sum
          bigram_prob = 0 # math.log10(epsi)
          trigram_prob = 0 # math.log10(epsi)
      else:
        # i >= 1
        one_before = sentence[i-1]
        bigram_prob = bigram.get((get_index(one_before), get_index(t_i)), -100)
        trigram_prob = 0 # math.log10(epsi*epsi)
        if i > 1:
          two_before = sentence[i-2]
          trigram_prob = trigram.get((get_index(two_before), get_index(one_before), get_index(t_i)), -100)

      # Interpolate + update
      prob = math.log10(l1 * (10**unigram_prob) + l2 * (10**bigram_prob) + l3 * (10**trigram_prob))
      total_likelihood += prob
  
  print(-(total_likelihood/len(observed_set)))
  return -(total_likelihood/len(observed_set))


# Optimization
initial_lambdas = [1/3, 1/3, 1/3]
constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]  # l1 + l2 + l3 = 1
bounds = [(0, 1), (0, 1), (0, 1)]  # Each lambda must be between 0 and 1

# Minimize the negative log-likelihood
result = minimize(negative_log_likelihood, initial_lambdas, args=None, 
                  constraints=constraints, bounds=bounds)

print(f"Success? : {result.success}")
print(f"Message : {result.message}")

# Optimal weights
optimal_lambdas = result.x
print("Optimal lambdas:", optimal_lambdas)

# [9.49650121e-01 9.37157253e-18 5.03498791e-02]

#testval = negative_log_likelihood(initial_lambdas)
#print(testval)