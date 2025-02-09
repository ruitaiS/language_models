import torch
import torch.nn as nn
from torch.nn import functional as F

class SHA(nn.Module): # Single Head Attention
	def __init__(self, d, d_k, d_v, use_mask = True):
		super().__init__()
		self.use_mask = use_mask
		self.W_Q = nn.Linear(d, d_k)
		self.W_K = nn.Linear(d, d_k)
		self.W_V = nn.Linear(d, d_v)
		#self.W_V = nn.Linear(d_v, d)
		print('sha init')

	def forward(self, X):
		Q = self.W_Q(X) # Queries Matrix
		K = self.W_K(X) # Keys Matrix
		V = self.W_V(X) # Values Matrix

		weights = F.softmax(self.mask(self.scaled_dot_prod(Q, K)), dim=-1)
		product = torch.matmul(weights, V) # (batch_size, seq_length, d_v)
		#output = self.W_0(product) # (batch_size, seq_length, d)
		print('sha forward')
		return product #output

	def mask (self, input):
		if not self.use_mask: return input

		batch_size, rows, cols = input.shape
		assert rows == cols, f"Matrix is not square: {rows}x{cols}"
		# TODO: Double check if this is flipped the right way
		tril = torch.tril(torch.ones(rows, rows, dtype=torch.bool))
		# print(tril)
		mask = tril.unsqueeze(0).expand(batch_size, -1, -1)
		return input.masked_fill(~mask, float('-inf'))

	def scaled_dot_prod(self, Q, K):
		batch_size, seq_len, d_k = Q.shape
		output = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
		# (batch_size,seq_len,seq_len)
		# Each token in a sequence is replaced with a vector showing attention scores for every other spot in the sequence
		return output 

class MHA(nn.Module): # Multi-Headed Attention
	def __init__(self, d, total_heads, use_mask = True):
		super().__init__()
		assert d % total_heads == 0, "d must be divisible by total number of heads"
		self.d_h = d // total_heads # instantiate SHAs with d_k and d_v set to d_h
		self.heads = nn.ModuleList([SHA(d, self.d_h, self.d_h, use_mask) for _ in range(total_heads)])
		self.W_0 = nn.Linear(d, d)
		print('mha init')
	def forward(self, X):
		output = torch.cat([head(X) for head in self.heads], dim=-1)
		output = self.W_0(output)
		print('mha forward')
		return output

class FFN(nn.Module):
	def __init__(self, d, d_hidden = None):
		super().__init__()
		# d_hidden is the number of nodes / dimensions on the hidden layer
		# (Book calls this dff)
		# W1, b1 take us to hidden layer
		# W2, b2 take us back to number of embed feature dimensions d
		# d >> dff >> d
		if not d_hidden or d_hidden <= d:
			d_hidden = 4*d

		self.W1 = nn.Parameter(torch.randn(d, d_hidden))
		self.b1 = nn.Parameter(torch.zeros(d_hidden))

		self.W2 = nn.Parameter(torch.randn(d_hidden, d))
		self.b2 = nn.Parameter(torch.zeros(d))
		print('ffn init')

	def forward(self, X):
		# ReLu(xW1+b1)W2 + b2
		hidden = torch.relu(torch.matmul(X, self.W1) + self.b1)
		output = torch.matmul(hidden, self.W2) + self.b2
		print('ffn forward')
		return output

class LayerNorm(nn.Module):
	def __init__(self, d):
		super().__init__()
		self.gain = nn.Parameter(torch.ones(d))
		self.offset = nn.Parameter(torch.zeros(d))
		print('layernorm init')
	def forward(self, X):
		# X.shape = (batch_size, seq_length, embedding dimension d)
		# Normalize each d-length embedding vector in X:
		means = X.mean(dim=-1, keepdim=True) # dim=-1 is d dimension
		sdevs = X.std(dim=-1, unbiased=False, keepdim=True)
		X = self.gain * ((X - means) / (sdevs + 1e-5)) + self.offset
		print('layernorm forward')
		return X

class TransformerBlock (nn.Module):
	def __init__(self, d, total_heads):
		super().__init__()
		self.norm1 = LayerNorm(d)
		self.norm2 = LayerNorm(d)
		self.ffn = FFN(d)
		self.mha = MHA(d, total_heads)
		print('transformer init')
	def forward(self, X):
		print(X)

		# pg. 10
		# with input x:
		residual = X
		X = self.norm1(X)
		X = self.mha(X)
		X += residual
		residual = X
		X = self.norm2(X)
		X = self.ffn(X)
		X += residual
		print('transformer forward')
		return X

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
		return X # X.shape = (batch_size, seq_len, d)
	
class LanguageModelHead(nn.Module):
	def __init__(self):
		super().__init__()
		# converts everything back
		# pg. 16-18
		# unembedding layer
		# softmax
		print('lm head init')
	def forward(self, X):
		print('lm head forward')
		return X

class LanguageModel(nn.Module):
	def __init__(self):
		
		print('todo')
		# embedding layer
		# blocks - set of TransformerBlock instances
		# Language model head
	def forward(self):
		print('todo')

#block_size = 8
#vocab_size = 100
#d = 32 # embedding dimensions