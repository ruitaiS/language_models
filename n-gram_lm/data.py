import pandas as pd

vocab = pd.read_csv('text/b0_vocab.txt', sep=' ', header=None,
                    quoting=3, # handle quotation marks
                    dtype={1: 'string'},  # Force the second column to be treated as string
                    keep_default_na=False # Disable default NaN detection ("None" won't be converted to NaN)
                    ).rename(columns={0: 'index', 1: 'word'})

unigram = pd.read_csv('text/b1_unigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_0', 1: 'e'})
bigram = pd.read_csv('text/b2_bigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_i', 1: 'x_j', 2:'e'})
trigram = pd.read_csv('text/b3_trigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_i', 1: 'x_j', 2: 'x_k', 3:'e'})

word_from_index = {index: vocab.at[index, 'word'] for index in vocab.index}
index_from_word = {vocab.at[index, 'word']: index for index in vocab.index}

def get_ifw():
  return index_from_word

def get_wfi():
  return word_from_index

def get_vocab():
  print(f'Vocab Length: {len(vocab)}')
  return vocab

def get_unigram():
  print(f'Unigram Length: {len(unigram)}')
  return unigram

def get_bigram():
  print(f'Bigram Length: {len(bigram)}')
  return bigram

def get_trigram():
  print(f'Trigram Length: {len(trigram)}')
  return trigram