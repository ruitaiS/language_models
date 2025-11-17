import os
import re
from bisect import bisect_right
from itertools import accumulate
import torch
from torch.utils.data import Dataset
from nltk.tokenize import RegexpTokenizer

#import nltk
#from nltk import sent_tokenize
#from nltk.corpus import brown
#nltk.data.path.append(os.path.dirname(os.path.dirname(__file__)))
base_path = os.path.dirname(os.path.abspath(__file__))

def preprocess_akjv(include_book=True):
    source_filepath = os.path.join('..', 'datasets', 'akjv.txt')
    # utf-8-sig strips leading BOM char
    with open(source_filepath, encoding="utf-8-sig") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]


    pattern = re.compile(r'^\s*(?P<book>.+?)\s+(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.+?)\s*$')
    processed_lines = []
    for line in lines:
        match = pattern.match(line)
        if not match:
            print(f"Line '{line}' discarded")
            continue
        if include_book:
            processed = f"{match['book']}\t{match['text']}"
        else:
            processed = f"{match['text']}"
        processed_lines.append(processed)
    return processed_lines

def tokenize(text, tokenization='char'):
    assert tokenization in ('char', 'word'), (
    f"tokenization must be 'char' or 'word' - got {tokenization}")

    if tokenization == 'word':
        tokenizer = RegexpTokenizer(r"\s+|<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
        tokens = tokenizer.tokenize(text)
        return tokens
    else: # tokenization == 'char'
        return list(text)

def build_and_encode(lines, tokenization='char'):
    print(f"Building vocabulary using {tokenization} tokenization...")
    vocab = sorted(set(tokenize("".join(lines), tokenization)))

    vocab.insert(0, '<?>') # out of dictionary token
    vocab.insert(1, '<s>') # start token
    vocab.insert(2, '</s>') # end token
    vocab.insert(3, '<>') # pad token

    vocab_size = len(vocab)
    idx2token = dict(enumerate(vocab))
    token2idx = {word:i for i, word in idx2token.items()}
    print(f"Final vocabulary size: {vocab_size}\n")

    print("Encoding Lines...")
    encoded_lines = []
    for line in lines:
        tokens = tokenize(line, tokenization)
        encoded = [token2idx.get(token, 0) for token in tokens]
        encoded_lines.append([1] + encoded + [2])
    print(f"Finished encoding {len(encoded_lines)} lines\n")
    return encoded_lines, vocab_size, idx2token, token2idx

# TODO: Split into lines and Encode to token idx's upstream of this
# TODO: Each line must be bracketed with start_token '<s>' and end_token '</s>'
# TODO: Batching logic occurs downstream
class TransformerDataset(Dataset):
    def __init__(self, encoded_lines, context_len,
                 start_token_idx=1, end_token_idx=2, pad_token_idx=3):
        assert len(encoded_lines) >= 1 and context_len >= 1
        assert all(
            len(line) > 2 and
            line[0] == start_token_idx and
            line[-1] == end_token_idx
            for line in encoded_lines
        )

        self.lines = encoded_lines
        self.context_len = context_len
        self.pad_token_idx = pad_token_idx

        counts = [len(line)-1 for line in self.lines]
        self.cumulative_counts = list(accumulate(counts))
        #print(self.cumulative_counts)


    def __getitem__(self, idx):
        assert (idx >=0 and idx < self.cumulative_counts[-1])

        line_idx = bisect_right(self.cumulative_counts, idx)
        p_0 = self.cumulative_counts[line_idx-1] if line_idx > 0 else 0

        x_f = idx - p_0 + 1 # +1 because slices don't include last index
        y_f = x_f + 1

        x_i = max(x_f - self.context_len, 0)
        y_i = max(y_f - self.context_len, 0)

        pad_x = max(self.context_len - x_f,0)
        pad_y = max(self.context_len - y_f,0)

        x = [self.pad_token_idx]*pad_x + self.lines[line_idx][x_i:x_f]
        y = [self.pad_token_idx]*pad_y + self.lines[line_idx][y_i:y_f]

        return x, y

    def __len__(self):
        return self.cumulative_counts[-1]

def batch(batch_size, context_len, dataset, shuffle=True):
    xft, _ = extract_aux(dataset)['vocab']
    dataset = [[xft.get(token, xft["<?>"]) for token in sentence] for sentence in dataset]

    num_samples = sum(len(sentence) - 1 for sentence in dataset)

    inputs = [None]*num_samples
    targets = [None]*num_samples

    insert_idx = 0
    for sentence in dataset:
        sequence = [xft['<>']] * (context_len - 1 ) + [xft['<s>']] 
        for token in sentence[1:]:
            inputs[insert_idx] = sequence.copy()
            next_sequence = sequence[1:] + [token]
            targets[insert_idx] = next_sequence
            sequence = next_sequence
            insert_idx += 1
    print(f"Sequences expected: {num_samples}, sequences created: {insert_idx}")

    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    #inputs = np.array(inputs)
    #targets = np.array(targets)
    # num_samples, context_len
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

    #input_batches = inputs.reshape(-1, batch_size, context_len)
    #target_batches = targets.reshape(-1, batch_size, context_len)
    input_batches = inputs.view(-1, batch_size, context_len)
    target_batches = targets.view(-1, batch_size, context_len)

    print(f"Input Batches: {input_batches.shape}")
    print(f"Target Batches: {target_batches.shape}")

    return input_batches, target_batches

