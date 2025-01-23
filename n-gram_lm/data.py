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

def process_csv(filepath, dType=None):
  output = {}
  with open(filepath, mode='r') as f:
    for row in csv.reader(f, delimiter=' '):
      if not dType:
        output[tuple(row[:-1])] = row[-1] # Last column is the value; other columns form a tuple acting as key
      else:
        output[tuple(row[:-1])] = dType(row[-1])
  return output    

def get_lookups():
  # TODO: These abbreviations are confusing and retarded
  # tfx >> token from index ; xft >> index from token
  tfx = process_csv('text/b0_vocab.txt')
  tfx = {a[0]:b for a, b in tfx.items()}
  xft = {b:a for a, b in tfx.items()}
  print(f"tfx dtype: {type(next(iter(tfx.values())))}")
  print(f"xft dtype: {type(next(iter(xft.values())))}")
  return xft, tfx

def get_unigram():
  unigram = process_csv('text/b1_unigram_counts.txt', dType=float)
  unigram = {a[0]:b for a, b in unigram.items()}
  print(f'Unigram Length: {len(unigram)}')
  print(f"Unigram dtype: {type(next(iter(unigram.values())))}")
  return unigram

def get_bigram():
  bigram = process_csv('text/b2_bigram_counts.txt', dType=float)
  print(f'Bigram Length: {len(bigram)}')
  print(f"Bigram dtype: {type(next(iter(bigram.values())))}")
  return bigram

def get_trigram():
  trigram = process_csv('text/b3_trigram_counts.txt', dType=float)
  print(f'Trigram Length: {len(trigram)}')
  print(f"Trigram dtype: {type(next(iter(trigram.values())))}")
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