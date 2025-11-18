import os
import pandas as pd
import numpy as np

import data
import generate

#model_name = 'trigram_imitation'
#model_data = load_model(model_name)
#context_len = model_data['params']['context_len']

tokenization='char'
include_book=True
if tokenization=='char':
    delimiter = ''
else:
    delimiter = ' '

processed_lines = data.preprocess_akjv(include_book)
encoded_lines, vocab_size, idx2token, token2idx = data.build_and_encode(processed_lines, tokenization)
print(f"Sample Line Encoded:\n{encoded_lines[0]}\n")
print(f"Sample Line Reconstructed:\n{delimiter.join([idx2token.get(idx, '<?>') for idx in encoded_lines[0]])}\n")

print("To exit, type 'q!' or 'quit', or press CTRL-C\n")
while True:
    try:
        raw_input = input(">> User Prompt: ")
    except:
        print("\nExiting..")
        break

    if raw_input.lower() in ['q!', 'quit']:
        break
    else:
        prompt_tokens = data.tokenize(raw_input, tokenization)
        prompt_indices = [token2idx.get(token, 0) for token in prompt_tokens]
        reconstructed = delimiter.join([idx2token.get(idx, "<?>") for idx in prompt_indices])

        print(f"\nTokenized Prompt: {prompt_tokens}")
        print(f"Prompt Token Indices: {prompt_indices}")
        print(f"Reconstructed Prompt: {reconstructed}\n")
        
        #print(f"\nGenerated Text:")

        #model = model_data['core']
        #output = model.generate(user_prompt, response_length = 100)
        #output = output[context_len:] # Drop the prompt out of the continuation
        #print(f"{' '.join(output)}\n")
