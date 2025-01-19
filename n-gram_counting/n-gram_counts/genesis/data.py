import pandas as pd

vocab = pd.read_csv('vocab.txt', sep=' ', header=None,
                    quoting=3, # handle quotation marks
                    dtype={1: 'string'},  # Force the second column to be treated as string
                    keep_default_na=False # Disable default NaN detection ("None" won't be converted to NaN)
                    ).rename(columns={0: 'index', 1: 'word'})

#index 988 is the word "None" which sometimes is converted to a NaN
#Uncomment this code to verify that it reads in correctly
#print("Check 988")
#test = vocab.iloc[985:996]  # View rows 985 to 995
#print(test['word'])

unigram = pd.read_csv('unigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_0', 1: 'e'})
bigram = pd.read_csv('bigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_i', 1: 'x_j', 2:'e'})
trigram = pd.read_csv('trigram_counts.txt', sep=' ', header=None).rename(columns={0: 'x_i', 1: 'x_j', 2: 'x_k', 3:'e'})

unigram['P'] = unigram['e'].apply(lambda e: 10**e)
bigram['P'] = bigram['e'].apply(lambda e: 10**e)
trigram['P'] = trigram['e'].apply(lambda e: 10**e)

#Normalize P Values, Ensure Sums to 1
unigram['subset_sum']= unigram.groupby(['x_0'])['P'].transform('sum')
unigram['normalized_P'] = unigram['P']/unigram['subset_sum']
bigram['subset_sum']= bigram.groupby(['x_i'])['P'].transform('sum')
bigram['normalized_P'] = bigram['P']/bigram['subset_sum']
trigram['subset_sum']= trigram.groupby(['x_i', 'x_j'])['P'].transform('sum')
trigram['normalized_P'] = trigram['P']/trigram['subset_sum']

#CS-ify the indices
vocab['index'] -= 1
unigram['x_0'] -= 1
bigram[['x_i', 'x_j']] -= 1
trigram[['x_i', 'x_j', 'x_k']] -= 1

bigram = bigram[['x_i', 'x_j', 'normalized_P']]
trigram = trigram[['x_i', 'x_j', 'x_k', 'normalized_P']]

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