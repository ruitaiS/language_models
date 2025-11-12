import os
import pandas as pd
import numpy as np
import math
import Levenshtein
from tokenizer import tokenize
import data
from model_handler import load_model
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


def logP_emission(observed, token_index, l = 0.01):
    k = Levenshtein.distance(observed.lower(), tfx.get(token_index, '<?>').lower())
    return k * math.log(l) - math.lgamma(k + 1) - l

def recurse(remaining, phrases, log_phrase_probs, tfx = None):
    log_phrase_probs = np.array(log_phrase_probs) # TODO: check if you can remove this
    observed = remaining.pop(0)
    log_emission_probs = np.array([logP_emission(observed, token_index, l=0.01) for token_index in range(vocab_size)])
    log_product_matrix = log_phrase_probs[:, None] + log_transition_matrix + log_emission_probs[None, :] # Explanation for this in ecse_526/p2.py line 35
    indices = np.argmax(log_product_matrix, axis=0).tolist()
    # TODO: Remove below after confirmed useless (feeds strings instead of token indexes as sequences)
    #updated_phrases = [phrases[phrase_index] + [tfx.get(token_index, '<?>')] for token_index, phrase_index in enumerate(indices)]
    updated_phrases = [phrases[phrase_index] + [token_index] for token_index, phrase_index in enumerate(indices)]
    updated_phrase_log_probs = np.max(log_product_matrix, axis=0)

    # TODO: sometimes it outputs '<s>' and idk why
    if tfx is not None: # eg. should print
        token_ids = updated_phrases[np.argmax(updated_phrase_log_probs)]
        phrase = ' '.join([tfx.get(token_id, '<?>') for token_id in token_ids])
        print(f"\r{phrase}", end="", flush=True)
        if not remaining: print("\n")

    if remaining:
        return recurse(remaining, updated_phrases, updated_phrase_log_probs, tfx)
    else:
        token_ids = updated_phrases[np.argmax(updated_phrase_log_probs)]
        return token_ids

#-------------------------------------------------------------------------------
model_name = 'trigram_imitation'
model_data = load_model(model_name)
xft, tfx = model_data['vocab']
context_len = model_data['params']['context_len']
vocab_size = len(xft)
print(f"Vocab Size: {vocab_size}")
bigram_lp = model_data['bigram']

log_transition_matrix = bigram_lp.pivot(index = 'x_i', columns = 'x_j', values='e').fillna(float('-inf'))
log_transition_matrix = log_transition_matrix.reindex(index=list(range(vocab_size)), columns=list(range(vocab_size)), method='pad', fill_value=float('-inf'))

print("\nType 'q!' or 'quit' to exit\n")
while True:
    try:
        raw_input = input(">> User Input: ")
    except:
        print("\nExiting..")
        break

    if raw_input.lower() in ['q!', 'quit']:
        break
    else:
        user_tokens = tokenize(raw_input)
        if len(user_tokens) == 0:
            user_token_ids_1 = [xft['<>']]*(context_len-1) + [xft['<s>']]
            user_token_ids_2 = [xft['<>']]*(context_len-1) + [xft['<s>']]
        else:
            # Direct Lookup With Fallbacks:
            user_token_ids_1 = [xft.get(token, xft.get(token.lower(), xft.get(token.upper(), xft.get('<?>')))) for token in user_tokens]
            print(f"\nMethod 1 (Direct Lookup):")
            print(f"{' '.join([tfx.get(token_id, '<???>') for token_id in user_token_ids_1])}")


            print(f"Method 2 (HMM Correction):")
            phrase_probs = pd.Series([0] * vocab_size)
            phrase_probs[xft['<s>']] = 1
            phrases = [[] for _ in range(vocab_size)]
            phrases[xft['<s>']] = [xft['<s>']]
            user_token_ids_2 = recurse(user_tokens, phrases, phrase_probs, tfx)
            #print(f"{' '.join([tfx.get(token_id, '<???>') for token_id in user_token_ids_2])}\n")

        corrected_outputs = [user_token_ids_1, user_token_ids_2]
        correction_method = 1
        print(f'Text continuation on method {correction_method} output:')
        user_prompt = corrected_outputs[correction_method-1]

        model = model_data['core']
        output = model.generate(user_prompt, response_length = 100)
        output = output[context_len:] # Drop the prompt out of the continuation
        print(f"{' '.join(output)}\n")
