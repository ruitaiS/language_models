import os
import nltk
import math
import pandas as pd

nltk.data.path.append(os.path.dirname(__file__))
#nltk.download('punkt_tab', download_dir=os.path.dirname(__file__))
from nltk.tokenize import word_tokenize

source_text = 'genesis'

unigrams = {}
bigrams = {}
trigrams = {}

with open('source_text/'+source_text+'/preprocessed.txt', "r") as infile:
    for line in infile:
        tokens = word_tokenize(line) # word_tokenize(line.lower())
        #print("Tokens:", tokens)
        for i in range(len(tokens)):
            if i == 0: # if first
                unigrams['<s>'] = unigrams.get('<s>', 0) + 1
                bigrams[('<s>',tokens[i])] = bigrams.get(('<s>',tokens[i]), 0) + 1            
                if i < len(tokens) - 1: # (at least one more)
                    trigrams[('<s>',tokens[i], tokens[i+1])] = trigrams.get(('<s>',tokens[i], tokens[i+1]), 0) + 1
                else: # (first and also last)
                    trigrams[('<s>',tokens[i], '</s>')] = trigrams.get(('<s>',tokens[i], '</s>'), 0) + 1
            else:
                unigrams[tokens[i]] = unigrams.get(tokens[i], 0) + 1
                if i == len(tokens) - 1: # if no more
                    bigrams[(tokens[i], '</s>')] = bigrams.get((tokens[i], '</s>'), 0) + 1
                    unigrams['</s>'] = unigrams.get('</s>', 0) + 1
                else: # at least one more
                    bigrams[(tokens[i], tokens[i+1])] = bigrams.get((tokens[i], tokens[i+1]), 0) + 1
                    if i == len(tokens) - 2: # (exactly one more)
                        trigrams[(tokens[i], tokens[i+1], '</s>')] = trigrams.get((tokens[i], tokens[i+1], '</s>'), 0) + 1
                    else: #  (more than one more)
                        trigrams[(tokens[i], tokens[i+1], tokens[i+2])] = trigrams.get((tokens[i], tokens[i+1], tokens[i+2]), 0) + 1

unigram_sums = sum(unigrams.values())
bigram_sums = sum(bigrams.values())
trigram_sums = sum(trigrams.values())

# Original counts files used log10, so keep that convention here
unigram_probs = {key: math.log10(count / unigram_sums) for key, count in unigrams.items()}
bigram_probs = {key: math.log10(count / bigram_sums) for key, count in bigrams.items()}
trigram_probs = {key: math.log10(count / trigram_sums) for key, count in trigrams.items()}

#print(len(unigram_probs))
#print(len(bigram_probs))
#print(len(trigram_probs))

unigram_df = pd.DataFrame(list(unigram_probs.items()), columns=['word', 'e']).sort_values(by='e', ascending=False).reset_index(drop=True)

bigram_df = pd.DataFrame(
    [(key[0], key[1], value) for key, value in bigram_probs.items()],
    columns=["n", "n+1", "e"]
).sort_values(by='e', ascending=False).reset_index(drop=True)

trigram_df = pd.DataFrame(
    [(key[0], key[1], key[2], value) for key, value in trigram_probs.items()],
    columns=["n", "n+1", "n+2", "e"]
).sort_values(by='e', ascending=False).reset_index(drop=True)

print(unigram_df)
print(bigram_df)
print(trigram_df)