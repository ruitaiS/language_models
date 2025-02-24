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
  tfx = {int(a[0]):b for a, b in tfx.items()} # TODO: This might be unnecessary; is a one element tuple real
  xft = {b:int(a) for a, b in tfx.items()}
  #print(f"tfx dtype: {type(next(iter(tfx.values())))}")
  #print(f"xft dtype: {type(next(iter(xft.values())))}")
  return xft, tfx

def get_training_sequences(batch_size, context_len, shuffle=True):
  '''
  One or more sequences per line
  should always capture the start of the line
  should end at end of line

  Let's save inter-line contexts for later exploration

  token list length = context length >> you can fit the entire sentence into the target without padding
  token list length - 1 = context length >> you capture front and end of the sentence without padding in either
  EX: context_len = 4

  sentence =  [<s>,   a, b, </s>]
  input =  [[<> , <>, <> , <s>], [<> ,  <>, <s>, a], [ <>, <s>, a,    b]]
  target = [[<> , <>, <s>,   a], [<> , <s>,   a, b], [<s>,   a, b, </s>]]

  input =  [[<s>, <>, <>, <>], [<> ,  <>, <s>, a], [ <>, <s>, a,    b]]
  target = [[  a, <>, <>, <>], [<> , <s>,   a, b], [<s>,   a, b, </s>]]

  len(sentence) = 4
  end_idx iterates from 1 to 3
  end_idx = 1:
  if (end_idx (1) - context_len (4) >= 0): False
    padding = context_len (4) - end_idx (1) = 3
    input = [<>, <>, <>] + sentence[0:1] = [<>, <>, <>, <s>]
    target = [<>, <>] + sentence[0:2] = [<>, <>, <s>, a]

  end_idx = 2:
  if (end_idx (2) - context_len (4) >= 0): False
    padding = context_len (4) - end_idx (2) = 2
    input = [<>, <>] + sentence[0:2] = [<>, <>, <s>, a]
    target = [<>] + sentence[0:3] = [<>, <s>, a, b]

  end_idx = 3:
  if (end_idx (3) - context_len (4) >= 0): False
    padding = context_len (4) - end_idx (3) = 1
    input = [<>] + sentence[0:3] = [<>, <s>, a, b]
    target = sentence[0:4] = [<s>, a, b, </s>]

  ---------------------------------------------

  sentence = [<s>, a, b, c, </s>]
  len(sentence) = 5
  end_idx iterates from 1 to 4
  end_idx = 1-3 are the same as the earlier sentence

  end_idx = 4:
  if (end_idx (4) - context_len (4) >= 0): True
    input = sentence[end_idx (4) - context_len (4) : end_idx]
    input = sentence[0:4] = [<s>, a, b, c]
    target = sentence[end_idx (4) - context_len (4) + 1 : end_idx + 1]
    target = sentence[1:5] = [a, b, c, </s>]


  end_idx =                 1,                 2,                3,                4
  input =  [[<>, <>, <>, <s>], [<>,  <>, <s>, a], [ <>, <s>, a, b], [<s>, a, b,   c ]]
  target = [[<>, <>, <s>,  a], [<>, <s>,   a, b], [<s>,   a, b,c ], [  a, b, c, </s>]]

----------------------------------------------

  sentence = [<s>, a, b, c, d, </s>]
  len(sentence) = 6
  end_idx iterates from 1 to 5

  end_idx = 5:
  if (end_idx (5) - context_len (4) >= 0): True
    input = sentence[end_idx (5) - context_len (4) : end_idx]
    input = sentence[1:5] = [a, b, c, d]
    target = sentence[end_idx (5) - context_len (4) + 1 : end_idx + 1]
    target = sentence[2:6] = [b, c, d, </s>]

  end_idx =                  1,                 2,                3,              4,              5
  input =  [[<>, <>,  <>, <s>], [<>,  <>, <s>, a], [ <>, <s>, a, b], [<s>, a, b, c], [a, b, c,    d]]
  target = [[<>, <>, <s>,   a], [<>, <s>,   a, b], [<s>,   a, b, c], [  a, b, c, d], [b, c, d, </s>]]
  '''
  xft, _ = get_vocab()
  train_set = get_train_set()
  train_set = [[xft.get(token, xft["<?>"]) for token in sentence] for sentence in train_set]
  num_samples = sum(len(sentence) - 1 for sentence in train_set)

  inputs = [None]*num_samples
  targets = [None]*num_samples
  print(f"Num Samples: {num_samples}")

  insert_idx = 0
  for sentence in train_set:
    for end_idx in range(1, len(sentence)):
      #print(f"end_idx: {end_idx}, {(end_idx - context_len >= 0)}, ")
      if (end_idx - context_len >= 0):
        inputs[insert_idx] = sentence[end_idx - context_len : end_idx]
        targets[insert_idx] = sentence[end_idx - context_len + 1 : end_idx + 1]
        #print(f"input = sentence[{end_idx - context_len} : {end_idx}]")
        #print(f"target = sentence[{end_idx - context_len + 1} : {end_idx + 1}]")
      else:
        padding = context_len - end_idx
        inputs[insert_idx] = [xft["<>"]] * padding + sentence[0:end_idx]
        targets[insert_idx] = [xft["<>"]] * (padding - 1) + sentence[0:end_idx + 1]
        #print(f"padding: {padding}")
        #print(f"input = {['<>']*padding} + sentence[0 : {end_idx}]")
        #print(f"target = {['<>']*(padding-1)} + sentence[0 : {end_idx+1}]")
      insert_idx += 1

  # num_samples, context_len
  inputs = torch.tensor(inputs, dtype=torch.long)
  targets = torch.tensor(targets, dtype=torch.long)
  print(f"Input Shape: {inputs.shape}")
  print(f"Target Shape: {targets.shape}")

  if shuffle:
    reorder = torch.randperm(len(inputs))
    inputs = inputs[reorder]
    targets = targets[reorder]

  remainder = num_samples % batch_size
  if remainder:
    inputs = inputs[:-remainder]
    targets = targets[:-remainder]
  
  input_batches = inputs.view(-1, batch_size, context_len)
  target_batches = targets.view(-1, batch_size, context_len)

  print(f"Input Batches: {input_batches.shape}")
  print(f"Target Batches: {target_batches.shape}")

  return input_batches, target_batches





def gts(batch_size, seq_len, shuffle= True):
  xft, _ = get_vocab()
  train_set = get_train_set()
  all_tokens = [token for sentence in train_set for token in sentence]
  print(all_tokens[:50])
  token_ids = [xft.get(token, xft["<?>"]) for token in all_tokens]
  
  num_samples = len(token_ids) - seq_len - 1

  inputs = torch.empty((num_samples, seq_len), dtype=torch.long)
  targets = torch.empty((num_samples, seq_len), dtype=torch.long)
  for start_idx in range(num_samples):
    inputs[start_idx] = torch.tensor(token_ids[start_idx : start_idx + seq_len], dtype=torch.long)
    targets[start_idx] = torch.tensor(token_ids[start_idx + 1 : start_idx + seq_len + 1], dtype=torch.long)

  if shuffle:
    reorder = torch.randperm(len(inputs))
    inputs = inputs[reorder]
    targets = targets[reorder]

  remainder = num_samples % batch_size
  if remainder:
    inputs = inputs[:-remainder]
    targets = targets[:-remainder]
  
  input_batches = inputs.view(-1, batch_size, seq_len)
  target_batches = targets.view(-1, batch_size, seq_len)

  print(f"Input Batches: {input_batches.shape}")
  print(input_batches)
  print(f"Target Batches: {target_batches.shape}")
  print(target_batches)

  return input_batches, target_batches


# Sample batch of data for testing purposes only
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
      start_idx = random.randint(0, len(token_ids) - seq_len - 2)
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