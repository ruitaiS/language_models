import os
import csv
import random
import torch
import nltk
#nltk.data.path.append(os.path.dirname(__file__))
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

def get_vocab():
  # These abbreviations are confusing and retarded (Well too bad)
  # tfx >> token from index ; xft >> index from token
  tfx = process_csv('text/b0_vocab.txt')
  tfx = {a[0]:b for a, b in tfx.items()} # TODO: This might be unnecessary; is a one element tuple real
  xft = {b:int(a) for a, b in tfx.items()}
  #print(f"tfx dtype: {type(next(iter(tfx.values())))}")
  #print(f"xft dtype: {type(next(iter(xft.values())))}")
  return xft, tfx

# Sample batch of data for testing purposes
def sample(batch_size, seq_len):
  xft, tfx = get_vocab()
  train_set = get_train_set()

  # Flatten the dataset for easier sequential sampling
  all_tokens = [token for sentence in train_set for token in sentence]
  
  # Convert tokens to indices
  token_ids = [xft.get(token, xft["<?>"]) for token in all_tokens]
  

  inputs = []
  targets = []

  for _ in range(batch_size):
      start_idx = random.randint(0, len(token_ids) - seq_len - 1)
      input_seq = token_ids[start_idx : start_idx + seq_len]
      target_seq = token_ids[start_idx + 1 : start_idx + seq_len + 1]
      inputs.append(input_seq)
      targets.append(target_seq)

  return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
  



def get_train_set():
  train_set = []
  with open("text/a1_train_set.txt", "r") as test_file:
    for line in test_file:
      parts = line.split("\t")
      if len(parts) > 1:
        text = parts[1].strip()
        tokens = word_tokenize(text)
        tokens = ["<s>"] + tokens + ["</s>"]
        train_set.append(tokens)
      else:
        print(f'Ignored line: {line}')
  return train_set

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