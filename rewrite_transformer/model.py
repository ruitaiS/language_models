import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, vocab_size, context_len):
        super().__init__()
        self.E = nn.Embedding(vocab_size,embedding_dim) # word embeddings
        self.P = nn.Embedding(context_len,embedding_dim) # position embeddings
        self.register_buffer("positions", torch.arange(context_len).unsqueeze(0))

    def forward(self, token_batch, seq_len):
        return self.E(token_batch) + self.P(self.positions[:, :seq_len])
    
class FFN(nn.Module): # Feed Forward Network
    def __init__(self, d, expansion_ratio):
        super().__init__()
        assert expansion_ratio>1, "FFN expansion ratio must produce a larger hidden dimension"

        self.network = nn.Sequential(
            nn.Linear(d, int(d*expansion_ratio)),
            nn.ReLU(), # TODO: toggle for nn.GeLU() # nn.SiLU()
            nn.Linear(int(d*expansion_ratio), d),
        )

    def forward(self, X):
        return self.network(X)

class TransformerBlock (nn.Module):
    # TODO: more intuitive dropout parameter passing
    def __init__(self, d, total_heads, ffn_expansion_ratio, attention_head_dropout, post_mha_dropout, post_ffn_dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(d, eps=1e-5, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d, eps=1e-5, elementwise_affine=True)
        self.dropout1 = nn.Dropout(p=post_mha_dropout)
        self.dropout2 = nn.Dropout(p=post_ffn_dropout)
        self.mha = MHA(d, total_heads, attention_head_dropout)
        self.ffn = FFN(d, ffn_expansion_ratio)

    def forward(self, X, attention_mask):
        # Note pre-norm architecture; slightly modernized from original 2017 paper
        # TODO: worth writing a little on why it's better this way vs. the original spec
        # mainly allows direct gradient flow through residuals

        residual = X
        X = self.norm1(X)
        X = self.mha(X, attention_mask)
        X = self.dropout1(X) + residual

        residual = X
        X = self.norm2(X)
        X = self.ffn(X)
        X = self.dropout2(X) + residual
        return X

class MHA(nn.Module): # Multi-Headed Attention
    def __init__(self, d, total_heads, attention_head_dropout):
        super().__init__()
        assert d % total_heads == 0, "d must be divisible by total number of heads"
        self.d_h = d // total_heads # instantiate SHAs with d_k and d_v set to d_h
        self.heads = nn.ModuleList([SHA(d, self.d_h, self.d_h, attention_head_dropout) for _ in range(total_heads)])
        self.W_0 = nn.Linear(d, d)
    def forward(self, X, attention_mask):
        output = torch.cat([head(X, attention_mask) for head in self.heads], dim=-1)
        output = self.W_0(output)
        return output

class SHA(nn.Module): # Single Head Attention
    # d_k and d_v also known as head dimension, d_h, in MHA context
    def __init__(self, d, d_k, d_v, attention_head_dropout):
        super().__init__()
        self.W_Q = nn.Linear(d, d_k)
        self.W_K = nn.Linear(d, d_k)
        self.W_V = nn.Linear(d, d_v)
        self.dropout = nn.Dropout(p=attention_head_dropout)
        self.scaling = 1/math.sqrt(d_k)

    def forward(self, X, attention_mask):
        Q = self.W_Q(X) # Queries Matrix (batch_size, seq_len, d_k)
        K = self.W_K(X) # Keys Matrix (batch_size, seq_len, d_k)
        V = self.W_V(X) # Values Matrix (batch_size, seq_len, d_v)

        scores = (Q @ K.transpose(-2, -1)) * self.scaling
        scores = scores.masked_fill(~attention_mask, float('-1e9'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        product = torch.matmul(weights, V)
        return product

class LanguageModel(nn.Module):
    def __init__(self, context_len, embedding_dim, num_layers, total_heads, vocab_size, pad_token_idx):
        super().__init__()

        # TODO import as params
        ffn_expansion_ratio=4
        embedding_dropout = 0.1
        post_mha_dropout=0.1
        post_ffn_dropout=0.1
        attention_head_dropout=0.1

        self.vocab_size = vocab_size
        self.context_len = context_len
        self.pad_token_idx = pad_token_idx
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.total_heads = total_heads
        self.ffn_expansion_ratio = ffn_expansion_ratio
        self.embedding_dropout = embedding_dropout
        self.post_mha_dropout = post_mha_dropout
        self.post_ffn_dropout = post_ffn_dropout
        self.attention_head_dropout = attention_head_dropout

        self.register_buffer("causal_mask", torch.tril(torch.ones(context_len, context_len, dtype=torch.bool)))
        self.embedding_layer = EmbeddingLayer(embedding_dim, vocab_size, context_len)
        self.dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embedding_dim, total_heads, ffn_expansion_ratio,
                              attention_head_dropout, post_mha_dropout, post_ffn_dropout)
                              for _ in range(num_layers)])

    def forward(self, token_batch):
        # Create attention mask
        # TODO: add cross sequence mask
        # sequence_mask = [...]
        _, seq_len = token_batch.shape
        padding_mask = (token_batch != self.pad_token_idx)
        query_mask = padding_mask[:, :, None]
        key_mask = padding_mask[:, None, :] 
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attention_mask = causal_mask & query_mask & key_mask

        # Pass Through Model:
        X = self.embedding_layer(token_batch, seq_len)
        X = self.dropout(X)
        for layer in self.transformer_layers:
            X = layer(X, attention_mask)
        logits = torch.matmul(X, self.embedding_layer.E.weight.T)
        return logits
    
    def device(self):
        return next(self.parameters()).device
    
    def params(self):
        return {
            "vocab_size": self.vocab_size,
            "context_len": self.context_len, 
            "pad_token_idx": self.pad_token_idx,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "total_heads": self.total_heads,
            "ffn_expansion_ratio": self.ffn_expansion_ratio,
            "embedding_dropout": self.embedding_dropout,
            "post_mha_dropout": self.post_mha_dropout,
            "post_ffn_dropout": self.post_ffn_dropout,
            "attention_head_dropout": self.attention_head_dropout
        }
