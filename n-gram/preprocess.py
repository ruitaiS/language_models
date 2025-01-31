import os
import random
import nltk
import math
import csv
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

#nltk.data.path.append(os.path.dirname(__file__))
nltk.data.path.append(os.path.dirname(os.path.dirname(__file__)))
from nltk.tokenize import word_tokenize

# TODO: (For more general text forms, split by sentence. Rn it's already one line per sentence so dw)
#from nltk.tokenize import sent_tokenize

# Might be interesting to explore sentence sequences. Maybe some groups of sentences go together

input_file = "text/akjv.txt"
unigram_counts = {}
bigram_counts = {}
trigram_counts = {}

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
        tokens = word_tokenize(text) #word_tokenize(text.lower())
        #print("Tokens:", tokens)
        for i in range(len(tokens)):
            unigram_counts[tokens[i]] = unigram_counts.get(tokens[i], 0) + 1
            if i == 0: # First token is the second character of a bigram starting with '<s>'
                unigram_counts['<s>'] = unigram_counts.get('<s>', 0) + 1
                bigram_counts[('<s>',tokens[i])] = bigram_counts.get(('<s>',tokens[i]), 0) + 1            
                if i < len(tokens) - 1: # (at least one more)
                    trigram_counts[('<s>',tokens[i], tokens[i+1])] = trigram_counts.get(('<s>',tokens[i], tokens[i+1]), 0) + 1
                else: # (first and also last)
                    trigram_counts[('<s>',tokens[i], '</s>')] = trigram_counts.get(('<s>',tokens[i], '</s>'), 0) + 1
            else:
                if i == len(tokens) - 1: # if no more
                    bigram_counts[(tokens[i], '</s>')] = bigram_counts.get((tokens[i], '</s>'), 0) + 1
                    unigram_counts['</s>'] = unigram_counts.get('</s>', 0) + 1
                else: # at least one more
                    bigram_counts[(tokens[i], tokens[i+1])] = bigram_counts.get((tokens[i], tokens[i+1]), 0) + 1
                    if i == len(tokens) - 2: # (exactly one more)
                        trigram_counts[(tokens[i], tokens[i+1], '</s>')] = trigram_counts.get((tokens[i], tokens[i+1], '</s>'), 0) + 1
                    else: #  (more than one more)
                        trigram_counts[(tokens[i], tokens[i+1], tokens[i+2])] = trigram_counts.get((tokens[i], tokens[i+1], tokens[i+2]), 0) + 1
    else:
        print(f'Unprocessed Line: {line}')

print(f"{len(unigram_counts)} unigrams in training set")
print(f"{len(bigram_counts)} bigrams in training set")
print(f"{len(trigram_counts)} trigrams in training set")

# Assign index mapping and create vocab hash
xft = {token: index + 1 for index, token in enumerate(sorted(unigram_counts.keys()))} # xft eg. index from token
vocab = {b:a for a,b in xft.items()} # tfx token from index

# Create Probabilities hash map
bigram_probs = {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}
trigram_probs = {trigram: count / bigram_counts[(trigram[0], trigram[1])] for trigram, count in trigram_counts.items()}

del unigram_counts['<s>'] # TODO: I think this is ok after the bigram / trigram probs are calculated.
del unigram_counts['</s>'] # makes sure the unigram isn't randomly dropping these into the phrase
unigram_sum = sum(unigram_counts.values())
unigram_probs = {unigram: count / unigram_sum for unigram, count in unigram_counts.items()}

# Normalize Probabilities
# Previous n-gram files (and the existing code to process them) use log10(P), so keep that convention here
# It's to make multiplication easier bc you can just add them as logs then exponentiate at the end
unigram_lp = {unigram: math.log10(prob) for unigram, prob in unigram_probs.items()}
bigram_lp = {bigram: math.log10(prob) for bigram, prob in bigram_probs.items()}
trigram_lp = {trigram: math.log10(prob) for trigram, prob in trigram_probs.items()}

# Convert counts hashes to all use indexes instead of token strings
unigram_lp = {xft[unigram] : logprob for unigram, logprob in unigram_lp.items()}
bigram_lp = {(xft[bigram[0]], xft[bigram[1]]): logprob for bigram, logprob in bigram_lp.items()}
trigram_lp = {(xft[trigram[0]], xft[trigram[1]], xft[trigram[2]]): logprob for trigram, logprob in trigram_lp.items()}

# TODO sort so humans can read too

with open('text/b0_vocab.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(sorted(vocab.items(), key= lambda item: item[1])) # item = (index, token); sort by token

with open('text/b1_unigram_counts.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    for x_i, e in unigram_lp.items():
        writer.writerow([x_i, e])

with open('text/b2_bigram_counts.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    for (x_i, x_j), e in bigram_lp.items():
        writer.writerow([x_i, x_j, e])

with open('text/b3_trigram_counts.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    for (x_i, x_j, x_k), e in trigram_lp.items():
        writer.writerow([x_i, x_j, x_k, e])