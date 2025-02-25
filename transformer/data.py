import os
import csv
import random
import torch
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

def get_vocab():
  # These abbreviations are confusing and retarded (Well too bad)
  # tfx >> token from index ; xft >> index from token
  tfx = process_csv(os.path.join(base_path, 'text/b0_vocab.txt'))
  tfx = {int(a[0]):b for a, b in tfx.items()} # TODO: This might be unnecessary; is a one element tuple real
  xft = {b:int(a) for a, b in tfx.items()}
  #print(f"tfx dtype: {type(next(iter(tfx.values())))}")
  #print(f"xft dtype: {type(next(iter(xft.values())))}")
  return xft, tfx

def get_sequences(batch_size, context_len, shuffle=True, dataset='training'):
  '''
  context_len = 4

  sentence = [<s>, a, b, c, d, </s>] # length = 6
  token_id =               <s>,                  a,                 b,                c,              d,           </s>
  input =  [[<>, <>,  <>,  <>], [<>, <>,  <>, <s>], [<>,  <>, <s>, a], [ <>, <s>, a, b], [<s>, a, b, c], [a, b, c,    d]]
  target = [[<>, <>,  <>, <s>], [<>, <>, <s>,   a], [<>, <s>,   a, b], [<s>,   a, b, c], [  a, b, c, d], [b, c, d, </s>]]
  '''
  xft, _ = get_vocab()
  fetch = {'training' : get_train_set(),
           'validation' : get_dev_set(),
           'test' : get_test_set()}
  dataset = fetch[dataset]
  dataset = [[xft.get(token, xft["<?>"]) for token in sentence] for sentence in dataset]
  num_samples = sum(len(sentence) - 1 for sentence in dataset)

  inputs = [None]*num_samples
  targets = [None]*num_samples
  print(f"Num Samples: {num_samples}")

  insert_idx = 0
  for sentence in dataset:
    sequence = [xft['<>']] * (context_len - 1 ) + [xft['<s>']]
    for token_id in sentence[1:]:
      
      inputs[insert_idx] = sequence.copy()
      next_sequence = sequence[1:] + [token_id]
      targets[insert_idx] = next_sequence
      #print(targets[insert_idx])
      sequence = next_sequence
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


# Sample batch of data for testing purposes only
def sample(batch_size, seq_len):
  xft, tfx = get_vocab()
  dataset = get_train_set()

  # Flatten the dataset for easier sequential sampling
  all_tokens = [token for sentence in dataset for token in sentence]
  
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
  dataset = []
  with open(os.path.join(base_path, 'text/a1_train_set.txt'), 'r') as test_file:
    for line in test_file:
      parts = line.split("\t")
      if len(parts) > 1:
        text = parts[1].strip()
        tokens = word_tokenize(text)
        tokens = ["<s>"] + tokens + ["</s>"]
        dataset.append(tokens)
      else:
        print(f'Ignored line: {line}')
  return dataset

def get_dev_set():
  dataset = []
  with open(os.path.join(base_path, 'text/a2_dev_set.txt'), 'r') as test_file:
    for line in test_file:
      parts = line.split("\t")
      if len(parts) > 1:
        text = parts[1].strip()
        tokens = word_tokenize(text)
        tokens = ["<s>"] + tokens + ["</s>"]
        dataset.append(tokens)
      else:
        print(f'Ignored line: {line}')
  return dataset

def get_test_set():
  dataset = []
  with open(os.path.join(base_path, 'text/a3_test_set.txt'), 'r') as test_file:
    for line in test_file:
      parts = line.split("\t")
      if len(parts) > 1:
        text = parts[1].strip()
        tokens = word_tokenize(text)
        tokens = ["<s>"] + tokens + ["</s>"]
        dataset.append(tokens)
      else:
        print(f'Ignored line: {line}')
  return dataset