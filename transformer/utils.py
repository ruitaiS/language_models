import torch
import torch.nn as nn
from torch.nn import functional as F

'''block_size = 8
vocab_size = 100'''

'''def mask (input_matrix):
	rows, cols = input_matrix.shape
	if rows != cols:
		raise ValueError(f"Matrix is not square: {rows}x{cols}")
	else:
		tril = torch.tril(torch.ones(rows, rows))
		return input_matrix.masked_fill(tril == 0, float('-inf'))

def scaled_dot_prod(Q, K):
	N, d_k = Q.shape
	return torch.matmul(Q, K.T) / (d_k ** 0.5)'''

N = 10
test_matrix = torch.zeros((N,N))
print(mask(test_matrix))

'''d = 32 # embedding vector dimensions
d_k = 64 # query\key vector dimensions
d_v = 64 # value vector dimensions (Usually d_v = d_k)'''

'''E = nn.Embedding(vocab_size, d) # Embedding table E
P = nn.Embedding(block_size, d) # Positional Embedding table P

indices = torch.tensor([5, 23, 99, 87, 56, 77, 12, 92]) # block_size length tensor of token indices
positions = torch.arange(len(indices))
token_embeddings = E(indices)
pos_embeddings = P(positions)
X = token_embeddings + pos_embeddings # Composite Embeddings (Word + position)'''

'''W_Q = nn.Linear(d, d_k)
W_K = nn.Linear(d, d_k)
W_V = nn.Linear(d, d_v)'''

'''Q = W_Q(X) # Queries Matrix
K = W_K(X) # Keys Matrix
V = W_V(X) # Values Matrix'''

# Feedforward Layer
# Layer Norm
# Residual Stream

