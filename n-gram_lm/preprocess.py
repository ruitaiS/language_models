import os
import random
import nltk
import math
import csv
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

nltk.data.path.append(os.path.dirname(__file__))
from nltk.tokenize import word_tokenize

# TODO: (For more general text forms, split by sentence. Rn it's already one line per sentence so dw)
#from nltk.tokenize import sent_tokenize

# Might be interesting to explore sentence sequences. Maybe some groups of sentences go together

input_file = "text/akjv.txt"
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
        tokens = word_tokenize(text) #word_tokenize(text.lower())
        #print("Tokens:", tokens)
        for i in range(len(tokens)):
            unigrams[tokens[i]] = unigrams.get(tokens[i], 0) + 1
            if i == 0: # First token is the second character of a bigram starting with '<s>'
                unigrams['<s>'] = unigrams.get('<s>', 0) + 1
                bigrams[('<s>',tokens[i])] = bigrams.get(('<s>',tokens[i]), 0) + 1            
                if i < len(tokens) - 1: # (at least one more)
                    trigrams[('<s>',tokens[i], tokens[i+1])] = trigrams.get(('<s>',tokens[i], tokens[i+1]), 0) + 1
                else: # (first and also last)
                    trigrams[('<s>',tokens[i], '</s>')] = trigrams.get(('<s>',tokens[i], '</s>'), 0) + 1
            else:
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

print(f"{len(unigrams)} unigrams in training set")
print(f"{len(bigrams)} bigrams in training set")
print(f"{len(trigrams)} trigrams in training set")

# Normalize Probabilities
# Previous n-gram files (and the existing code to process them) use log10(P), so keep that convention here
# It's to make multiplication easier bc you can just add them as logs then exponentiate at the end
unigrams = {x: math.log10(count / unigram_sums) for x, count in unigrams.items()}
bigrams = {x: math.log10(count / bigram_sums) for x, count in bigrams.items()}
trigrams = {x: math.log10(count / trigram_sums) for x, count in trigrams.items()}

# Assign index mapping and create vocab hash
xft = {token: index + 1 for index, token in enumerate(sorted(unigrams.keys()))} # xft eg. index from token
vocab = {b:a for a,b in xft.items()} # tfx token from index

# Convert counts hashes to all use indexes instead of token strings
unigrams = {xft[unigram] : logprob for unigram, logprob in unigrams.items()}
bigrams = {(xft[bigram[0]], xft[bigram[1]]): logprob for bigram, logprob in bigrams.items()}
trigrams = {(xft[trigram[0]], xft[trigram[1]], xft[trigram[2]]): logprob for trigram, logprob in trigrams.items()}

# TODO sort so humans can read too

with open('text/b0_vocab.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(sorted(vocab.items(), key= lambda item: item[1])) # item = (index, token); sort by token

with open('text/b1_unigram_counts.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    for x_i, e in unigrams.items():
        writer.writerow([x_i, e])

with open('text/b2_bigram_counts.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    for (x_i, x_j), e in bigrams.items():
        writer.writerow([x_i, x_j, e])

with open('text/b3_trigram_counts.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    for (x_i, x_j, x_k), e in trigrams.items():
        writer.writerow([x_i, x_j, x_k, e])