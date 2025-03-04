import os
import pandas as pd 
import numpy as np
import math
import Levenshtein
from tokenizer import tokenize
import data

'''
Defaulting to lower case, then uppercase behavior
xft.get(token, xft.get(token.lower(), xft.get(token.upper(), '<?>')))

TODO / TO *try* :
- Expand contractions if they can't be found, eg let's >> let us; that's >> that is
- Contextually does it learn 's as "us" when after "let" but "is" after "that"?

It does this with the '<s>' some times but not other times and idk why:
Partial Phrase: ['surely']
Partial Phrase: ['<s>', 'Surely', 'there']
Partial Phrase: ['<s>', 'Surely', 'there', 'is']
Partial Phrase: ['<s>', 'Surely', 'there', 'is', 'come']
  Final Phrase: ['<s>', 'Surely', 'there', 'is', 'come', 'away']
'''

xft, tfx = data.get_vocab()
vocab_size = len(xft)
# TODO below:
bigram_lp = data.get_bigram()

log_transition_matrix = bigram_lp.pivot(index = 'x_i', columns = 'x_j', values='e').fillna(float('-inf'))
log_transition_matrix = log_transition_matrix.reindex(index=list(range(vocab_size)), columns=list(range(vocab_size)), method='pad', fill_value=float('-inf'))

def logP_emission(observed, token_index, l = 0.01):
	k = Levenshtein.distance(observed.lower(), tfx.get(token_index, '<?>').lower())
	return k * math.log(l) - math.lgamma(k + 1) - l

def recurse(remaining, phrases, log_phrase_probs, iter=0):
	#print(f'Iteration: {iter}')

	log_phrase_probs = np.array(log_phrase_probs) # TODO: check if you can remove this
	observed = remaining.pop(0)
	log_emission_probs = np.array([logP_emission(observed, token_index, l=0.01) for token_index in range(vocab_size)])
	log_product_matrix = log_phrase_probs[:, None] + log_transition_matrix + log_emission_probs[None, :] # Explanation for this in ecse_526/p2.py line 35
	indices = np.argmax(log_product_matrix, axis=0).tolist()
	updated_phrases = [phrases[phrase_index] + [tfx.get(token_index, '<?>')] for token_index, phrase_index in enumerate(indices)]
	updated_phrase_log_probs = np.max(log_product_matrix, axis=0)
	# TODO: sometimes it outputs '<s>' and idk why
	if remaining:
		print(f'Partial Phrase: {updated_phrases[np.argmax(updated_phrase_log_probs)]}')
		return recurse(remaining, updated_phrases, updated_phrase_log_probs, iter = iter + 1)
	else:
		corrected_phrase = updated_phrases[np.argmax(updated_phrase_log_probs)]
		print(f'  Final Phrase: {corrected_phrase}\n')
		return corrected_phrase

print("\nType your input after '>>>'")
print("Type ':q' or ':quit' to exit\n")
while True:
	#
	raw_input = input(">> ")
	if raw_input.lower() in [':q', ':quit']:
		break
	else:
		user_tokens = tokenize(raw_input)

		# Direct Lookup With Fallbacks:
		user_token_ids = [xft.get(token, xft.get(token.lower(), xft.get(token.upper(), xft.get('<?>')))) for token in user_tokens]
		print(f"Direct Lookup: {' '.join([tfx.get(token_id, '<???>') for token_id in user_token_ids])}\n")

		phrase_probs = pd.Series([0] * vocab_size)
		phrase_probs[xft['<s>']] = 1
		phrases = [[] for _ in range(vocab_size)]
		phrases[xft['<s>']] = ['<s>']
		corrected = recurse(user_tokens, phrases, phrase_probs)
