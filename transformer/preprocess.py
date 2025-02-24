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

input_file = "../datasets/akjv.txt"
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
    else:
        print(f'Unprocessed Line: {line}')
unigram_counts['<s>'] = 0
unigram_counts['</s>'] = 0
unigram_counts['<?>'] = 0
unigram_counts['<>'] = 0


# Assign index mapping and create vocab hash
xft = {token: index for index, token in enumerate(sorted(unigram_counts.keys()))} # xft eg. index from token
vocab = {b:a for a,b in xft.items()} # tfx token from index

print(f"{len(vocab)} words in training set vocab")


with open('text/b0_vocab.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(sorted(vocab.items(), key= lambda item: item[1])) # item = (index, token); sort by token