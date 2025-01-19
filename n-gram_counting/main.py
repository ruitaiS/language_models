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
            if i == 0: # first
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


# Create Probabilities hash map
# Previous n-gram files (and the existing code to process them) use log10(P), so keep that convention here
unigram_sums = sum(unigrams.values())
bigram_sums = sum(bigrams.values())
trigram_sums = sum(trigrams.values())
unigram_probs = {x: math.log10(count / unigram_sums) for x, count in unigrams.items()}
bigram_probs = {x: math.log10(count / bigram_sums) for x, count in bigrams.items()}
trigram_probs = {x: math.log10(count / trigram_sums) for x, count in trigrams.items()}

# Convert to dataframes
unigram_df = pd.DataFrame(list(unigram_probs.items()), columns=['x_0', 'e']).sort_values(by='x_0')
bigram_df = pd.DataFrame(
    [(x[0], x[1], e) for x, e in bigram_probs.items()],
    columns=["x_i", "x_j", "e"]
).sort_values(by='x_i')
trigram_df = pd.DataFrame(
    [(x[0], x[1], x[2], e) for x, e in trigram_probs.items()],
    columns=["x_i", "x_j", "x_k", "e"]
).sort_values(by='x_i')

# Create token vocabulary
vocab_df = unigram_df['x_0'].sort_values().reset_index(drop=True).to_frame()
vocab_df['id'] = vocab_df.index + 1

# TODO: Optimize
# Create mappings to id numbers
# Replace token strings with id numbers
id_map = dict(zip(vocab_df['x_0'], vocab_df['id']))
unigram_df['x_0'] = unigram_df['x_0'].replace(id_map)
bigram_df['x_i'] = bigram_df['x_i'].replace(id_map)
bigram_df['x_j'] = bigram_df['x_j'].replace(id_map)
trigram_df['x_i'] = trigram_df['x_i'].replace(id_map)
trigram_df['x_j'] = trigram_df['x_j'].replace(id_map)
trigram_df['x_k'] = trigram_df['x_k'].replace(id_map)


#TODO Decide Directory Structure
vocab_outfile = 'n-gram_counts/'+source_text+'/vocab.txt'
unigram_outfile = 'n-gram_counts/'+source_text+'/unigram_counts.txt'
bigram_outfile = 'n-gram_counts/'+source_text+'/bigram_counts.txt'
trigram_outfile = 'n-gram_counts/'+source_text+'/trigram_counts.txt'

vocab_df[['id', 'x_0']].to_csv(vocab_outfile, index=False, header=False, sep=' ')
unigram_df[['x_0', 'e']].to_csv(unigram_outfile, index=False, header=False, sep=' ')
bigram_df[['x_i', 'x_j', 'e']].to_csv(bigram_outfile, index=False, header=False, sep=' ')
trigram_df[['x_i', 'x_j', 'x_k', 'e']].to_csv(trigram_outfile, index=False, header=False, sep=' ')


    



