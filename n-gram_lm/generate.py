import pandas as pd
import numpy as np
import data

vocab = data.get_vocab()
unigram = data.get_unigram()
bigram = data.get_bigram()
trigram = data.get_trigram()
index_from_word = data.get_ifw()
word_from_index = data.get_wfi()

#def word(index):
  #return vocab.at[index, 'word']

#print(vocab)
#print(unigram)
#print(bigram)
print(trigram)