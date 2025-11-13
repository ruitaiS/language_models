import os
import pandas as pd
import numpy as np
import math
import Levenshtein
from tokenizer import tokenize
import data
from model_handler import load_model

model_name = 'trigram_imitation'
model_data = load_model(model_name)
xft, tfx = model_data['vocab']
context_len = model_data['params']['context_len']
vocab_size = len(xft)
print(f"Vocab Size: {vocab_size}")

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
        prompt_tokens = utils.tokenize(raw_input)
        prompt_indices = [1] + [token2idx.get(token, 0) for token in prompt_tokens]
        reconstructed = [idx2token.get(idx, "()") for idx in prompt_indices]

        print(f"Tokenized Input: {prompt_tokens}")
        print(f"Input Token Indices: {prompt_indices}")
        print(f"Reconstructed: {reconstructed}")
        
        print(f"\nGenerated Text:")

        #model = model_data['core']
        #output = model.generate(user_prompt, response_length = 100)
        #output = output[context_len:] # Drop the prompt out of the continuation
        #print(f"{' '.join(output)}\n")
