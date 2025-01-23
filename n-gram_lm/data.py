import os
#import pandas as pd
import csv
import nltk
nltk.data.path.append(os.path.dirname(__file__))
from nltk.tokenize import word_tokenize

# TODO: There was something weird about "None" becoming NaN with pandas; 
# Check if you need the same workaround with hashes
'''
vocab = pd.read_csv('text/b0_vocab.txt', sep=' ', header=None,
                    quoting=3, # handle quotation marks
                    dtype={1: 'string'},  # Force the second column to be treated as string
                    keep_default_na=False # Disable default NaN detection ("None" won't be converted to NaN)
                    ).rename(columns={0: 'index', 1: 'token'})
'''

def process_csv(filepath):
  output = {}
  with open(filepath, mode='r') as f:
    for row in csv.reader(f, delimiter=' '):
      output[tuple(row[:-1])] = row[-1] # Last column is the value; other columns form a tuple acting as key
  return output    

def get_lookups():
  token_from_index = process_csv('text/b0_vocab.txt')
  index_from_token = {token: index for index, token in token_from_index.items()}
  return index_from_token, token_from_index

def get_unigram():
  unigram = process_csv('text/b1_unigram_counts.txt')
  print(f'Unigram Length: {len(unigram)}')
  return unigram

def get_bigram():
  bigram = process_csv('text/b2_bigram_counts.txt')
  print(f'Bigram Length: {len(bigram)}')
  return bigram

def get_trigram():
  trigram = process_csv('text/b3_trigram_counts.txt')
  print(f'Trigram Length: {len(trigram)}')
  return trigram

def get_dev_set():
  dev_set = []
  with open("text/a2_dev_set.txt", "r") as test_file:
    for line in test_file:
      parts = line.split("\t")
      if len(parts) > 1:
        text = parts[1].strip()
        tokens = word_tokenize(text)
        tokens = ["<s>"] + tokens + ["</s>"]
        dev_set.append(tokens)
      else:
        print(f'Ignored line: {line}')
  return dev_set

def get_test_set():
  test_set = []
  with open("text/a3_test_set.txt", "r") as test_file:
    for line in test_file:
      parts = line.split("\t")
      if len(parts) > 1:
        text = parts[1].strip()
        tokens = word_tokenize(text)
        tokens = ["<s>"] + tokens + ["</s>"]
        test_set.append(tokens)
      else:
        print(f'Ignored line: {line}')
  return test_set