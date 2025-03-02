import os
import numpy as np
import math
import Levenshtein
from tokenizer import tokenize
import data

xft, tfx = data.get_vocab()
vocab_size = len(xft)
# TODO below:
bigram = data.get_bigram()
transition_matrix = bigram.pivot(index = 'x_i', columns = 'x_j', values='normalized_P').fillna(0)
transition_matrix = transition_matrix.reindex(index=list(range(vocab_size)), columns=list(range(vocab_size)), method='pad', fill_value=0)

def logP_emission(observed, token_index, l = 0.01):
	k = Levenshtein.distance(observed.lower(), tfx.get(token_index, '<?>').lower())
	return k * math.log(l) - math.lgamma(k + 1) - l

def recurse(remaining, phrases, log_phrase_probs):
	log_phrase_probs = np.array(log_phrase_probs) # TODO: check if you can remove this
	observed = remaining.pop(0)
	log_emission_probs = np.array([logP_emission(observed, token_index, l=0.01) for token_index in range(len(xft))])
	log_product_matrix = log_phrase_probs[:, None] + log_transition_matrix + log_emission_probs[None, :] # Explanation for this in ecse_526/p2.py line 35
	indices = np.argmax(log_product_matrix, axis=0).tolist()
	updated_phrases = [phrases[phrase_index] + " " + tfx.get(token_index, '<?>') for token_index, phrase_index in enumerate(indices)]
	updated_phrase_log_probs = np.max(log_product_matrix, axis=0)
	if remaining:
		return recurse(remaining, updated_phrases, updated_phrase_log_probs)
	else:
		corrected_phrase = updated_phrases[np.argmax(updated_phrase_log_probs)]
		print(f'Corrected Phrase: {corrected_phrase}')
		return corrected_phrase

	

# TODO Intro blurb / whatever, ending with VV
while True:
	print("User Input")
	print("':q' or ':quit' to exit")
	raw_input = input(">> ")
	if raw_input.lower() in [':q', ':quit']:
		break
	else:
		user_tokens = tokenize(raw_input)
		user_token_ids = [xft.get(token, '<?>') for token in user_tokens] 

	print(f"Raw input as list: {user_tokens}")
	print(f"Input as tokens: {user_token_ids}")
	print(f"Token list expressed as a string: {' '.join([tfx.get(token_id, '???') for token_id in user_token_ids])}")
