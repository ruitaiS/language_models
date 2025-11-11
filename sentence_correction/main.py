import utils
import pandas as pd

vocab_size, idx2token, token2idx, log_transition_matrix = utils.extract_components()

print("To exit, type 'q!' or 'quit', or press CTRL-C\n")
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
        
        print(f"\nCorrected Text:")
        phrase_probs = pd.Series([0] * vocab_size)
        phrase_probs[token2idx['<s>']] = 1
        phrases = [[] for _ in range(vocab_size)]
        phrases[token2idx['<s>']] = [token2idx['<s>']]
        utils.recurse(input_tokens, phrases, phrase_probs, log_transition_matrix, idx2token)