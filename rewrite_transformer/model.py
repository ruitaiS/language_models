import torch
import torch.nn as nn
from torch.nn import functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, vocab_size, context_len):
        super().__init__()
        self.E = nn.Embedding(vocab_size,embedding_dim) # word embeddings
        self.P = nn.Embedding(context_len,embedding_dim) # position embeddings
        self.register_buffer("positions", torch.arange(context_len).unsqueeze(0))

    def forward(self, token_batch):
        # Accepts one batch of tokens, shape (batch_size, seq_len), where seq_len <= context_len
        # tokens = torch.tensor([1, 2, 3, 4], [5, 6, 7, 8]) >> tokens.shape = (2, 4)

        _, seq_len = token_batch.shape # TODO: The shape can be passed down, so you're not re-deriving shape every batch
        X = self.E(token_batch) + self.P(self.positions[:, :seq_len]) # Composite Embeddings (Word + position)
        #print(f"embedding_layer(tokens): {X}")

        # X.shape = (batch_size, seq_len,embedding_dim)
        return X

class LayerNorm(nn.Module):
    # TODO: You might want to just use nn.LayerNorm
    def __init__(self, d):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.offset = nn.Parameter(torch.zeros(d))
        #print('layernorm init')
    def forward(self, X):
        # X.shape = (batch_size, seq_length, embedding dimension d)
        # Normalize each d-length embedding vector in X:
        means = X.mean(dim=-1, keepdim=True) # dim=-1 is d dimension
        sdevs = X.std(dim=-1, unbiased=False, keepdim=True)
        X = self.gain * ((X - means) / (sdevs + 1e-9)) + self.offset
        #print(f'layernorm(X): {X}')
        return X
    
class FFN(nn.Module): # Feed Forward Network
    # TODO: You might want to simplify this to a nn.Sequential(linear, relu, linear)
    # you're rolling your own linear layers here with manual weights and biases lol

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
        #print('ffn init')

    def forward(self, X):
        # ReLu(xW1+b1)W2 + b2
        hidden = torch.relu(torch.matmul(X, self.W1) + self.b1) # ReLu(xW1 + b1)
        output = torch.matmul(hidden, self.W2) + self.b2 # hidden * W2 + b2
        #print(f'ffn(X): {output}')
        return output

class TransformerBlock (nn.Module):
    # TODO: dropout
    def __init__(self, d, total_heads):
        super().__init__()

        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)
        self.ffn = FFN(d)
        
        # TODO: Make sure it works with these library ones first
        '''
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear,
            nn.ReLU,
            nn.Linear)
        '''

        self.mha = MHA(d, total_heads)
        #print('transformer init')
    def forward(self, X, attention_mask):
        # pg. 10
        # with input x:
        residual = X
        X = self.norm1(X)
        X = self.mha(X, attention_mask)
        X += residual
        residual = X
        X = self.norm2(X)
        X = self.ffn(X)
        X += residual
        #print(f"transformer(X) : {X}")
        #print(f"Transformer Block Output Shape: {X.shape}")
        return X

class MHA(nn.Module): # Multi-Headed Attention
    def __init__(self, d, total_heads):
        super().__init__()
        assert d % total_heads == 0, "d must be divisible by total number of heads"
        self.d_h = d // total_heads # instantiate SHAs with d_k and d_v set to d_h
        self.heads = nn.ModuleList([SHA(d, self.d_h, self.d_h) for _ in range(total_heads)])
        self.W_0 = nn.Linear(d, d)
        #print('mha init')
    def forward(self, X, attention_mask):
        output = torch.cat([head(X, attention_mask) for head in self.heads], dim=-1)
        output = self.W_0(output)
        #print(f'mha(X): {output}')
        return output

class SHA(nn.Module): # Single Head Attention
    # d_k and d_v also known as head dimension, d_h, in MHA context
    def __init__(self, d, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d, d_k)
        self.W_K = nn.Linear(d, d_k)
        self.W_V = nn.Linear(d, d_v)
        self.d_k = d_k
        #self.W_0 = nn.Linear(d_v, d)
        #print('sha init')

    def forward(self, X, attention_mask):
        #print(f"SHA Forward X.shape: {X.shape}")
        # X.shape = (batch_size, seq_len, d)
        Q = self.W_Q(X) # Queries Matrix (batch_size, seq_len, d_k)
        K = self.W_K(X) # Keys Matrix (batch_size, seq_len, d_k)
        V = self.W_V(X) # Values Matrix (batch_size, seq_len, d_v)

        input = self.scaled_dot_prod(Q, K).masked_fill(~attention_mask, float('-1e9'))
        weights = F.softmax(input, dim=-1)
        product = torch.matmul(weights, V) # (batch_size, seq_length, d_v)
        #print('sha forward')

        # MHA class contains its own W_0 which aggregates across the attention heads
        # Uncomment below if doing explicitly single headed attention
        #output = self.W_0(product) # (batch_size, seq_length, d)
        #output

        #print(f"SHA Forward product.shape: {product.shape}")

        return product

    def scaled_dot_prod(self, Q, K):
        #print(f"Q.shape: {Q.shape}")
        #batch_size, seq_len, d_k = Q.shape
        output = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # (batch_size,seq_len,seq_len)
        # Each token in a sequence is replaced with a vector showing attention scores for every other spot in the sequence
        # Masking drops the scores for tokens that come after the current one
        return output

class LanguageModel(nn.Module):
    def __init__(self, context_len, embedding_dim, num_layers, total_heads, vocab_size, pad_token_idx):
        super().__init__()

        # TODO: These need to be stored when saving the model so it can be re-instantiated
        # context_len is the only one that really needs to be read off the model (used by generation function to truncate the input)
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.pad_token_idx = pad_token_idx
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.total_heads = total_heads
        self.register_buffer("causal_mask", torch.tril(torch.ones(context_len, context_len, dtype=torch.bool)))

        self.embedding_layer = EmbeddingLayer(embedding_dim, vocab_size, context_len)
        self.transformer_layers = nn.ModuleList([TransformerBlock(embedding_dim, total_heads) for _ in range(num_layers)])

    def forward(self, token_batch):
        # Accepts one batch of tokens of shape (batch_size, seq_len), where seq_len <= context_len
        #token_batch = np.array(token_batch.tolist()) # Should already be np.array out of the data.batch()
        #token_batch = np.vectorize(lambda token: self.token2idx.get(token, self.token2idx['<?>']))(token_batch)
        #token_batch = torch.tensor(token_batch, dtype=torch.long)

        # View Tokens flowing through:
        #for seq in token_batch:
        #    view = [self.idx2token[token.item()] for token in seq]
        #    print(' '.join(view))
        #print('')

        # Create attention mask
        # TODO: add cross sequence mask
        # sequence_mask = [...]

        _, seq_len = token_batch.shape
        padding_mask = (token_batch != self.pad_token_idx)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attention_mask = causal_mask & padding_mask[:, None, :]

        X = self.embedding_layer(token_batch)
        for layer in self.transformer_layers:
            X = layer(X, attention_mask)
        logits = torch.matmul(X, self.embedding_layer.E.weight.T)
        #print(f"Logits shape: {logits.shape}")
        return logits
