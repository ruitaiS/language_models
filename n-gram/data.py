import os
#import pandas as pd
import csv
import nltk
base_path = os.path.dirname(os.path.abspath(__file__))
nltk.data.path.append(os.path.dirname(os.path.dirname(__file__)))
from nltk.tokenize import word_tokenize

def process_csv(filepath, dType=None):
  # Converts csv to hash where last column is the value
  # Other columns form a tuple acting as key
  output = {}
  with open(filepath, mode='r') as f:
    for row in csv.reader(f, delimiter=' '):
      if not dType:
        output[tuple(row[:-1])] = row[-1]
      else:
        output[tuple(row[:-1])] = dType(row[-1])
  return output    

def get_lookups():
  # These abbreviations are confusing and retarded (Well too bad)
  # tfx >> token from index ; xft >> index from token
  tfx = process_csv(os.path.join(base_path, 'text/b0_vocab.txt'))
  tfx = {a[0]:b for a, b in tfx.items()} # TODO: This might be unnecessary; is a one element tuple real
  xft = {b:a for a, b in tfx.items()}
  #print(f"tfx dtype: {type(next(iter(tfx.values())))}")
  #print(f"xft dtype: {type(next(iter(xft.values())))}")
  return xft, tfx

def get_unigram():
  unigram = process_csv(os.path.join(base_path, 'text/b1_unigram_counts.txt'), dType=float)
  unigram = {a[0]:b for a, b in unigram.items()} # TODO: This might be unnecessary
  #print(f'Unigram Length: {len(unigram)}')
  #print(f"Unigram dtype: {type(next(iter(unigram.values())))}")
  return unigram

def get_bigram():
  bigram = process_csv(os.path.join(base_path, 'text/b2_bigram_counts.txt'), dType=float)
  #print(f'Bigram Length: {len(bigram)}')
  #print(f"Bigram dtype: {type(next(iter(bigram.values())))}")
  return bigram

def get_trigram():
  trigram = process_csv(os.path.join(base_path, 'text/b3_trigram_counts.txt'), dType=float)
  #print(f'Trigram Length: {len(trigram)}')
  #print(f"Trigram dtype: {type(next(iter(trigram.values())))}")
  return trigram

def get_dev_set():
  dev_set = []
  with open(os.path.join(base_path, 'text/a2_dev_set.txt'), 'r') as test_file:
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
  with open(os.path.join(base_path, 'text/a3_test_set.txt'), 'r') as test_file:
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