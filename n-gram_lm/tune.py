import data
import math
from scipy.optimize import minimize

unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
xft, tfx = data.get_lookups()
observed_set = data.get_dev_set()

def get_index(t_i):
  if t_i in xft:
    return xft[t_i]
  else:
    return -1

# Loss Function:
def negative_log_likelihood(params):
  l1, l2, l3, fallback = params
  ##print(f"l1: {params[0]}, l2: {params[1]}, l3: {params[2]}, fallback: {params[3]}")
  set_likelihood = 0
  not_in_vocab = 0

  for sentence in observed_set:
    sentence_likelihood = 0
    for i, t_i in enumerate(sentence):
      x_i = get_index(t_i)
      if x_i == -1:
        #print(f"Word {t_i} not in vocab")
        not_in_vocab += 1
        raw_prob = fallback
      else:
        if i == 1: # First word in the sentence after <s>
          bigram_prob = bigram.get((xft['<s>'], x_i), fallback)
          raw_prob = l2*(10**bigram_prob)
        else:
          x_i_min_1 = get_index(sentence[i-1])
          x_i_min_2 = get_index(sentence[i-2])
          unigram_prob = unigram.get(x_i, fallback)
          bigram_prob = bigram.get((x_i_min_1, x_i), fallback)
          trigram_prob = trigram.get((x_i_min_2, x_i_min_1, x_i), fallback)
          if int(x_i) <= 10: # Ignore unigram probs for punctuation
            raw_prob = l2 * (10**bigram_prob) + l3 * (10**trigram_prob)
          else:
            raw_prob = l1 * (10**unigram_prob) + l2 * (10**bigram_prob) + l3 * (10**trigram_prob)
      log_prob = math.log10(max(raw_prob, 1e-10))
      sentence_likelihood += log_prob
    set_likelihood += sentence_likelihood
  ##print(f"P: {total_likelihood/len(observed_set)}, Not In Vocab: {not_in_vocab}")
  return -(set_likelihood/len(observed_set))


# Optimization - Minimize NLL
initial_params = [1/3, 1/3, 1/3, -100]
constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:3]) - 1}]  # l1 + l2 + l3 = 1
bounds = [(0, 1), (0, 1), (0, 1), (-1000, 0)]
result = minimize(negative_log_likelihood, initial_params, 
                  constraints=constraints, bounds=bounds)

print(f"Success? : {result.success}")
print(f"Message : {result.message}")

optimal_lambdas = result.x
print(f"Optimal lambdas: {optimal_lambdas[:3]}, optimal fallback: {optimal_lambdas[3]}")