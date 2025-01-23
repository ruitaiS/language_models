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
  total_likelihood = 0

  # TODO: I think epsi gets doubled (if there's no bigram, there's obviously no trigram)


  for sentence in observed_set:
    for i, t_i in enumerate(sentence): # Let's adopt convention: t_i for token string representation, x_i for vocab index representation
      unigram_prob = unigram.get(get_index(t_i), math.log10(epsi))

      if i == 0:
          bigram_prob = math.log10(epsi) # TODO idk about this
          trigram_prob = math.log10(epsi)
      else:
        # i >= 1
        one_before = sentence[i-1]

        #print(f"last i: {get_index(one_before)}, last t_i: {one_before}")
        bigram_prob = bigram.get((get_index(one_before), get_index(t_i)), math.log10(epsi))
        #print(f"Unigram P: {unigram_prob} , Bigram P: {bigram_prob}, Trigram P: {trigram_prob}")
        trigram_prob = math.log10(epsi*epsi)
        if i > 1:
          two_before = sentence[i-2]
          trigram_prob = trigram.get((get_index(two_before), get_index(one_before), get_index(t_i)), math.log10(epsi))
          print(f"{two_before} {one_before} {t_i}")
          #print(f"Unigram P: {unigram_prob} , Bigram P: {bigram_prob}, Trigram P: {trigram_prob}")
      
      


      '''unigram_prob = unigram.get(get_index[t_i], epsi)

      if i == 0:
        bigram_prob = 0  # Strictly zero and not epsilon bc it is the first word
      else:
        bigram_prob = 
        
      if i == 1:
        trigram_prob = 0  # Strictly zero and not epsilon bc it is the first two words
      else:
        trigram_prob = trigram.get((sentence[i-2], sentence[i-1], word), epsi)
      '''
      print(f"Bigram prob: {bigram_prob}")
      if (bigram_prob > math.log10(epsi)):
        print(f"i: {i}, Phrase: {one_before}{t_i}")
      #if ( trigram_prob > math.log10(epsi)):
      #  print(f"i: {i}, Phrase: {two_before}{one_before}{t_i}")
      
      #else:
        #print(unigram_prob)
      #print(f"Unigram P: {unigram_prob} , Bigram P: {bigram_prob}, Trigram P: {trigram_prob}")
      # Interpolate + update
      #prob = l1 * unigram_prob #+ l2 * bigram_prob + l3 * trigram_prob
      #total_likelihood += prob
  return total_likelihood


# Optimization
initial_lambdas = [1/3, 1/3, 1/3]
#constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]  # l1 + l2 + l3 = 1
#bounds = [(0, 1), (0, 1), (0, 1)]  # Each lambda must be between 0 and 1

# Minimize the negative log-likelihood
#result = minimize(negative_log_likelihood, initial_lambdas, args=(ngram_probs, observed_counts), 
#                  constraints=constraints, bounds=bounds)

# Optimal weights
#optimal_lambdas = result.x
#print("Optimal lambdas:", optimal_lambdas)

testval = negative_log_likelihood(initial_lambdas)
print(testval)