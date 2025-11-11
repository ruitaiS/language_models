import utils
import pandas as pd

#vocab_size, idx2token, token2idx = utils.build_vocab()
vocab_size, idx2token, token2idx, log_transition_matrix = utils.extract_components()
print(list(idx2token.items())[:10])


#bigram_lp = None # TODO
#log_transition_matrix = bigram_lp.pivot(index = 'x_i', columns = 'x_j', values='e').fillna(float('-inf'))
#log_transition_matrix = log_transition_matrix.reindex(index=list(range(vocab_size)), columns=list(range(vocab_size)), method='pad', fill_value=float('-inf'))

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
        input_tokens = utils.tokenize(raw_input)
        input_indices = [1] + [token2idx.get(token, 0) for token in input_tokens]
        reconstructed = [idx2token.get(idx, "()") for idx in input_indices]

        print(f"Tokenized Input: {input_tokens}")
        print(f"Input Token Indices: {input_indices}")
        print(f"Reconstructed: {reconstructed}")

        
        print(f"Corrected Sentence:")
        phrase_probs = pd.Series([0] * vocab_size)
        phrase_probs[token2idx['<s>']] = 1
        phrases = [[] for _ in range(vocab_size)]
        phrases[token2idx['<s>']] = [token2idx['<s>']]
        corrected_token_ids = utils.recurse(input_tokens, phrases, phrase_probs, log_transition_matrix, idx2token)
        print(f"Corrected: {corrected_token_ids}")
