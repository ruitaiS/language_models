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

class LanguageModel(nn.Module):
	def __init__(self):
		print('todo')
	def forward(self):
		print('todo')