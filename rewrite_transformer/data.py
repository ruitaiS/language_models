import os
from itertools import accumulate
import torch
from torch.utils.data import Dataset
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer

base_path = os.path.dirname(os.path.abspath(__file__))

def tokenize(text, tokenization='char'):
    assert tokenization in ('char', 'word', 'bpe'), (
    f"tokenization must be 'char', 'word', or 'bpe' - got {tokenization}")

    if tokenization == 'word':
        tokenizer = RegexpTokenizer(r"<[^>\s]+>|[A-Za-z0-9]+'[A-Za-z0-9]+|\w+|[^\w\s]")
        words = tokenizer.tokenize(text)
        return words
    elif tokenization == 'char':
        return list(text)
    else: # tokenization == 'bpe'
        print("TODO: BPE")
        return None

def build_vocab(tokenization='char'):
    print(f"Building vocabulary using {tokenization} tokenization...")
    vocab = set()
    for fileid in brown.fileids():
        tokens = tokenize(brown.raw(fileid), tokenization)
        vocab.update(tokens)
    vocab = sorted(vocab)

    vocab.insert(0, '<?>') # out of dictionary token
    vocab.insert(1, '<s>') # start token
    vocab.insert(2, '</s>') # end token
    vocab.insert(3, '<>') # pad token

    vocab_size = len(vocab)
    idx2token = dict(enumerate(vocab))
    token2idx = {word:i for i, word in idx2token.items()}
    print(f"Finished, with final vocabulary size: {vocab_size}\n")
    return vocab_size, idx2token, token2idx

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

        # Don't think i actually need these two other than for verification
        #self.start_token_idx = start_token_idx
        #self.end_token_idx = end_token_idx

        counts = [len(line)-1 for line in self.lines]
        self.cumulative_counts = list(accumulate(counts))


    def __getitem__(self, idx):
        return None # TODO; see notes in readme

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

