import os
import csv
import random
import torch
from tokenizer import tokenize

base_path = os.path.dirname(os.path.abspath(__file__))

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
  dataset = get_dataset('train')
  vocab = set(['<s>', '</s>', '<?>', '<>'])
  for line in dataset:
      # print(f"Words in line: {len(line)}")
      vocab.update(line)
  # Assign index mapping and create vocab hash
  xft = {token : index for index, token in enumerate(sorted(vocab))} # index from token
  tfx = {index : token for index, token in enumerate(sorted(vocab))}# token from index

  print(f"{len(vocab)} words in training set vocab")

  with open(os.path.join(base_path, 'text/b0_vocab.txt'), 'w') as f:
      writer = csv.writer(f, delimiter=' ')
      writer.writerows(tfx.items()) # item = (index, token); sort by token
  return xft, tfx

def get_sequences(batch_size, context_len, shuffle=True, dataset='train'):
  '''
  context_len = 4

  sentence = [<s>, a, b, c, d, </s>] # length = 6
  token_id =               <s>,                  a,                 b,                c,              d,           </s>
  input =  [[<>, <>,  <>,  <>], [<>, <>,  <>, <s>], [<>,  <>, <s>, a], [ <>, <s>, a, b], [<s>, a, b, c], [a, b, c,    d]]
  target = [[<>, <>,  <>, <s>], [<>, <>, <s>,   a], [<>, <s>,   a, b], [<s>,   a, b, c], [  a, b, c, d], [b, c, d, </s>]]
  '''
  xft, _ = get_vocab()
  dataset = get_dataset(dataset)
  dataset = [[xft.get(token, xft["<?>"]) for token in sentence] for sentence in dataset]
  num_samples = sum(len(sentence) - 1 for sentence in dataset)

  inputs = [None]*num_samples
  targets = [None]*num_samples

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
  print(f"Sequences expected: {num_samples}, sequences created: {insert_idx}")

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
    
def get_dataset(dataset_name):
  dataset = []
  with open(os.path.join(base_path, f'text/a1_{dataset_name}_set.txt'), 'r') as data_file:
    for line in data_file:
      # print(line)
      tokens = tokenize(line)
      tokens = ["<s>"] + tokens + ["</s>"]
      dataset.append(tokens)
  return dataset
