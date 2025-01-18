import os
import nltk
nltk.data.path.append(os.path.dirname(__file__))
#nltk.download('punkt_tab', download_dir=os.path.dirname(__file__))
from nltk.tokenize import word_tokenize

source_text = 'genesis'

unigrams = {}
bigrams = {}
trigrams = {}

with open('source_text/'+source_text+'/preprocessed.txt', "r") as infile:
    for line in infile:
        tokens = word_tokenize(line.lower())
        print("Tokens:", tokens)
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



#print(unigrams)
#print("Unique Unigrams: " + str(len(unigrams)))
#print(unigrams['<s>'])
#print(unigrams['</s>'])

#print(bigrams)
print(trigrams)


# TODO: remember data.py does this transformation, so give this format
'''
unigram = pd.read_csv('unigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_0', 1: 'e'})
bigram = pd.read_csv('bigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_i', 1: 'x_j', 2:'e'})
trigram = pd.read_csv('trigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_i', 1: 'x_j', 2: 'x_k', 3:'e'})

unigram['P'] = unigram['e'].apply(lambda e: 10**e)
bigram['P'] = bigram['e'].apply(lambda e: 10**e)
trigram['P'] = trigram['e'].apply(lambda e: 10**e)

'''