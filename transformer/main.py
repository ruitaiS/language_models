import torch
import torch.nn as nn
from torch.nn import functional as F

'''
d = 32 # embedding vector dimensions
d_k = 64 # query\key vector dimensions
d_v = 64 # value vector dimensions (Usually d_v = d_k)

for mha:
d_k == d_v == head_size == d / num_heads
'''

class SHA(nn.Module): # Single Head Attention
	def __init__(self, d, d_k, d_v):
		super().__init__()
		self.W_Q = nn.Linear(d, d_k)
		self.W_K = nn.Linear(d, d_k)
		self.W_V = nn.Linear(d, d_v)

	def forward(self, X):
		Q = self.W_Q(X) # Queries Matrix
		K = self.W_K(X) # Keys Matrix
		V = self.W_V(X) # Values Matrix

		weights = F.softmax(self.mask(self.scaled_dot_prod(Q, K)), dim=-1)
		return torch.matmul(weights, V)

	def mask (self, input_matrix):
		rows, cols = input_matrix.shape
		if rows != cols:
			raise ValueError(f"Matrix is not square: {rows}x{cols}")
		else:
			tril = torch.tril(torch.ones(rows, rows))
			return input_matrix.masked_fill(tril == 0, float('-inf'))

	def scaled_dot_prod(self, Q, K):
		N, d_k = Q.shape
		return torch.matmul(Q, K.T) / (d_k ** 0.5)

class MHA(nn.Module): # Multi-Headed Attention
	def __init__(self):
		print('todo')
	def forward(self):
		print('todo')

class FFN():
	def __init__(self):
		print('todo')
	def forward(self):
		# ReLu(xW1+b1)W2 + b2
		# (whatever tf that means)
		# pg. 9
		print('todo')

class LayerNorm():
	def __init__(self):
		print('todo')
	def forward(self):
		# pg. 9
		print('todo')

class TransformerBlock (nn.Module):
	def __init__(self):
		print('todo')
	def forward(self):
		# pg. 10
		# with input x:
		# residual = x
		# x = LayerNorm(x)
		# x = MHA(x)
		# x += residual
		# residual = x
		# x = LayerNorm(x)
		# x = FFN(x)
		# x += residual
		print('todo')

class EmbeddingLayer(nn.Module):
	def __init__(self, d, vocab_size, block_size):
		super().__init__()
		self.E = nn.Embedding(vocab_size, d) # Embedding table E
		self.P = nn.Embedding(block_size, d) # Positional Embedding table P

	def forward(self, tokens):
		# accepts one batch of token_ids; tokens.shape >> (batch_size, seq_len)
		# tokens = torch.tensor([1, 2, 3, 4], [5, 6, 7, 8]) >> tokens.shape = (2, 4)

		batch_size, seq_len = tokens.shape
		positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1) # >> torch.tensor([0,1,2,3], [0,1,2,3])
		
		X = self.E(tokens) + self.P(positions) # Composite Embeddings (Word + position)
		return X

class LanguageModel(nn.Module):
	def __init__(self):
		
		print('todo')
		# embedding layer
		# blocks - set of TransformerBlock instances
		# 
	def forward(self):
		print('todo')

block_size = 8
vocab_size = 100
d = 32 # embedding dimensions