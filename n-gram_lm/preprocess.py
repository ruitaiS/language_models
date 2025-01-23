import os
import random
import nltk
import math
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

nltk.data.path.append(os.path.dirname(__file__))
from nltk.tokenize import word_tokenize

# TODO: (For more general text forms, split by sentence. Rn it's already one line per sentence so dw)
#from nltk.tokenize import sent_tokenize

# Might be interesting to explore sentence sequences. Maybe some groups of sentences go together

input_file = "text/a0_input.txt"
unigrams = {}
bigrams = {}
trigrams = {}

with open(input_file, "r") as infile:

    # Training / Dev / Test Split
    props = (8, 1, 1) # (Train, Dev, Test) normalized
    props = tuple(p / sum(props) for p in props)

    lines = infile.readlines()
    random.shuffle(lines)
    train_set = lines[:int(len(lines)*props[0])]
    dev_set = lines[int(len(lines)*props[0]):int(len(lines)*props[0]) + int(len(lines)*props[1])]
    test_set = lines[int(len(lines)*props[0]) + int(len(lines)*props[1]):]

    with open("text/a1_train_set.txt", "w") as train_file:
        print(f"{len(train_set)} lines in training set.")
        train_file.writelines(train_set)

    with open("text/a2_dev_set.txt", "w") as dev_file:
        print(f"{len(dev_set)} lines in dev set.")
        dev_file.writelines(dev_set)

    with open("text/a3_test_set.txt", "w") as test_file:
        print(f"{len(test_set)} lines in test set.")
        test_file.writelines(test_set)

for line in train_set:
    parts = line.split("\t")
    if len(parts) > 1:
        text = parts[1].strip()
        tokens = word_tokenize(text) # word_tokenize(line.lower())
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
    else:
        print(f'Unprocessed Line: {line}')

# Create Probabilities hash map
unigram_sums = sum(unigrams.values())
bigram_sums = sum(bigrams.values())
trigram_sums = sum(trigrams.values())

print(f"{len(unigrams)} unigrams")
print(f"{len(bigrams)} bigrams")
print(f"{len(trigrams)} trigrams")

# Normalize Probabilities
# Previous n-gram files (and the existing code to process them) use log10(P), so keep that convention here
# It's to make multiplication easier bc you can just add them as logs then exponentiate at the end
unigram_probs = {x: math.log10(count / unigram_sums) for x, count in unigrams.items()}
bigram_probs = {x: math.log10(count / bigram_sums) for x, count in bigrams.items()}
trigram_probs = {x: math.log10(count / trigram_sums) for x, count in trigrams.items()}


# TODO: Just use a hash; pandas is really slow and unnecessary
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

# Create mappings to id numbers
# Replace token strings with id numbers
id_map = dict(zip(vocab_df['x_0'], vocab_df['id']))
unigram_df['x_0'] = unigram_df['x_0'].replace(id_map)
bigram_df['x_i'] = bigram_df['x_i'].replace(id_map)
bigram_df['x_j'] = bigram_df['x_j'].replace(id_map)
trigram_df['x_i'] = trigram_df['x_i'].replace(id_map)
trigram_df['x_j'] = trigram_df['x_j'].replace(id_map)
trigram_df['x_k'] = trigram_df['x_k'].replace(id_map)


vocab_df[['id', 'x_0']].to_csv('text/b0_vocab.txt', index=False, header=False, sep=' ')
unigram_df[['x_0', 'e']].to_csv('text/b1_unigram_counts.txt', index=False, header=False, sep=' ')
bigram_df[['x_i', 'x_j', 'e']].to_csv('text/b2_bigram_counts.txt', index=False, header=False, sep=' ')
trigram_df[['x_i', 'x_j', 'x_k', 'e']].to_csv('text/b3_trigram_counts.txt', index=False, header=False, sep=' ')
