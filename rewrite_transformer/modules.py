import torch
import torch.nn as nn
from torch.nn import functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, d, vocab_size, context_len, padding_token_index):
        super().__init__()
        self.E = nn.Embedding(vocab_size, d) # Embedding table E
        self.P = nn.Embedding(context_len, d) # Positional Embedding table P
        self.padding_token_index = padding_token_index
        #print('embedding layer init')

    def forward(self, token_batch):
        # Accepts one batch of tokens, shape (batch_size, seq_len)
        # seq_len <= context_len
        # tokens = torch.tensor([1, 2, 3, 4], [5, 6, 7, 8]) >> tokens.shape = (2, 4)

        batch_size, seq_len = token_batch.shape
        positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1) # >> torch.tensor([0,1,2,3], [0,1,2,3])
        padding_mask = (token_batch != self.padding_token_index) # keep Trues, mask Falses
        X = self.E(token_batch) + self.P(positions) # Composite Embeddings (Word + position)
        #print(f"embedding_layer(tokens): {X}")

        # X.shape = (batch_size, seq_len, d) ; padding_mask.shape = (batch_size, seq_len)
        return X, padding_mask

class TransformerBlock (nn.Module):
    # TODO: dropout
    def __init__(self, d, total_heads, masked = True):
        super().__init__()
        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)
        self.ffn = FFN(d)
        self.mha = MHA(d, total_heads, masked)
        #print('transformer init')
    def forward(self, X, padding_mask):
        # pg. 10
        # with input x:
        residual = X
        X = self.norm1(X)
        X = self.mha(X, padding_mask)
        X += residual
        residual = X
        X = self.norm2(X)
        X = self.ffn(X)
        X += residual
        #print(f"transformer(X) : {X}")
        return X
    
class LanguageModelHead(nn.Module):
    def __init__(self, E):
        super().__init__()
        # for the unembedding matrix you can use the embedding matrix transposed
        # converts a d-length embedding back into a vocab_size length raw scores vector
        # softmax along the vector for a probability distribution
        # pg. 16-18

        # NOTE: Wonder what happens if you initialize it at E_t
        # but then trained it seperately
        self.register_buffer("E_t", E.weight.T)
        #print('lm head init')
    def forward(self, X):
        # X shape = (batch_size, seq_len, d)
        logits = torch.matmul(X, self.E_t) # shape = (batch_size, seq_len, vocab_size)

        # get shape (batch_size, seq_len, vocab_size) list of raw scores (logits) for each batch

        #print(f'lmh logits: {logits}')
        #print('lm head forward')
        return logits # shape (batch_size, seq_len, vocab_size)
    
class LayerNorm(nn.Module):
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

class MHA(nn.Module): # Multi-Headed Attention
    def __init__(self, d, total_heads, masked = True):
        super().__init__()
        assert d % total_heads == 0, "d must be divisible by total number of heads"
        self.d_h = d // total_heads # instantiate SHAs with d_k and d_v set to d_h
        self.heads = nn.ModuleList([SHA(d, self.d_h, self.d_h, masked) for _ in range(total_heads)])
        self.W_0 = nn.Linear(d, d)
        #print('mha init')
    def forward(self, X, padding_mask):
        output = torch.cat([head(X, padding_mask) for head in self.heads], dim=-1)
        output = self.W_0(output)
        #print(f'mha(X): {output}')
        return output

class SHA(nn.Module): # Single Head Attention
    # d_k and d_v also known as head dimension, d_h, in MHA context
    def __init__(self, d, d_k, d_v, masked = True):
        super().__init__()
        self.masked = masked
        self.W_Q = nn.Linear(d, d_k)
        self.W_K = nn.Linear(d, d_k)
        self.W_V = nn.Linear(d, d_v)
        #self.W_0 = nn.Linear(d_v, d)
        #print('sha init')

    def forward(self, X, padding_mask):
        # X.shape = (batch_size, seq_len, d)
        Q = self.W_Q(X) # Queries Matrix (batch_size, seq_len, d_k)
        K = self.W_K(X) # Keys Matrix (batch_size, seq_len, d_k)
        V = self.W_V(X) # Values Matrix (batch_size, seq_len, d_v)

        weights = F.softmax(self.mask(self.scaled_dot_prod(Q, K), padding_mask), dim=-1)
        product = torch.matmul(weights, V) # (batch_size, seq_length, d_v)
        #print('sha forward')

        # MHA class contains its own W_0 which aggregates across the attention heads
        # Uncomment below if doing explicitly single headed attention
        #output = self.W_0(product) # (batch_size, seq_length, d)
        #output

        return product

    def scaled_dot_prod(self, Q, K):
        batch_size, seq_len, d_k = Q.shape
        output = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        # (batch_size,seq_len,seq_len)
        # Each token in a sequence is replaced with a vector showing attention scores for every other spot in the sequence
        # Masking drops the scores for tokens that come after the current one
        return output

    # TODO: Double check if this is flipped the right way
    def mask (self, input, padding_mask):
        if not self.masked: return input

        batch_size, rows, cols = input.shape
        assert rows == cols, f"Matrix is not square: {rows}x{cols}"

        expanded_padding_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
        autoregression_mask = torch.tril(torch.ones(rows, rows, dtype=torch.bool)).repeat(batch_size, 1, 1)

        '''print(f"Padding Mask Shape: {padding_mask.shape}")
        print(padding_mask)
        print(f"Expanded Padding Mask Shape: {expanded_padding_mask.shape}")
        print(expanded_padding_mask)
        print(f"Autoregression Mask Shape: {autoregression_mask.shape}")
        print(autoregression_mask)'''

        return input.masked_fill(~expanded_padding_mask | ~autoregression_mask, float('-1e9'))
        #return input.masked_fill(~autoregression_mask, float('-1e9'))

class LanguageModel(nn.Module):
    def __init__(self, vocab, d, context_len, num_layers, total_heads):
        super().__init__()
        self.xft, self. tfx = vocab
        self.vocab_size = len(self.xft)
        self.context_len = context_len

        self.embedding_layer = EmbeddingLayer(d, self.vocab_size, context_len, padding_token_index = self.xft['<>'])
        self.transformer_layers = nn.ModuleList([TransformerBlock(d, total_heads) for _ in range(num_layers)])
        self.lm_head = LanguageModelHead(self.embedding_layer.E)
        
        self.loss_func = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.xft['<>']) # TODO: This needs a test

    def forward(self, token_batch, targets = None):
        # Accepts one batch of tokens of shape (batch_size, seq_len)
        # longer sequences truncated to context length
        # shorter sequences preserved
        token_batch = token_batch[:, -self.context_len:]
        #token_batch = np.array(token_batch.tolist()) # Should already be np.array out of the data.batch()
        #token_batch = np.vectorize(lambda token: self.xft.get(token, self.xft['<?>']))(token_batch)
        #token_batch = torch.tensor(token_batch, dtype=torch.long)

        # View Tokens flowing through:
        #for seq in token_batch:
        #    view = [self.tfx[token.item()] for token in seq]
        #    print(' '.join(view))
        #print('')


        X, padding_mask = self.embedding_layer(token_batch)
        for layer in self.transformer_layers:
            X = layer(X, padding_mask)
        logits = self.lm_head(X)
        #print(f"Logits shape: {logits.shape}")
        if targets is not None:
            # View Tokens flowing through:
            #for seq in targets:
            #    view = [self.tfx[token.item()] for token in seq]
            #    print(' '.join(view))
            #print('')

        #    print(f"Targets Shape: {targets.shape}")
            #targets = np.vectorize(lambda token: self.xft.get(token, self.xft['<?>']))(targets)
            #targets = torch.tensor(targets, dtype=torch.long)
            loss = self.calculate_loss(logits, targets)
        else:
            loss = None
        return logits, loss

    # Wrapper for self.loss_func mostly
    def calculate_loss(self, logits, targets):
        # In each batch:
        # logits shape (batch_size, seq_len, vocab_size)
        # targets shape (batch_size, seq_len)
        # Think of these as two batch_size, seq_len matrices
        # logits contains a vocab length logits vector in each entry
        # while targets simply contains a token index
        # The logits vectors from one matrix are trying to predict the corresponding token indices in the other.

        # For nn.CrossEntropyLoss:
        # We want to unroll these into two batch_size * seq_len lists, so that:
        # Each element of flattened_logits is a logits vector
        # Each element of flattened_targets is a token index
        flattened_logits = logits.view(-1, logits.shape[-1])
        flattened_targets = targets.view(-1)
        return self.loss_func(flattened_logits, flattened_targets)

    def generate(self, prompt= [], response_length=100):
        def sample(probabilities):
            # TODO See ch 10; top-k should be easy
            #print(f"Token Probabilities Shape: {probabilities.shape}")
            return torch.distributions.Categorical(probs=probabilities).sample()

            #return torch.argmax(probabilities, dim =-1)
        def next_token(token_batch):
            logits, _ = self.forward(token_batch)
            logits = logits[:, -1:, :] # Only want last token, not whole sequence
            #print(f"Logits shape: {logits.shape}")
            # dim=-1 >> softmax along vocab indices to get probabilities
            probabilities = F.softmax(logits, dim=-1)
            #print(f'lm probabilities: {probabilities}')
            return sample(probabilities)

        def batch_to_str(token_batch, display=False):
            output = [self.tfx[token.item()] for token in token_batch.squeeze(0)]
            if display: print(' '.join(output))
            return output


        # starting batch with batch_size = 1, seq_len = 1
        # Everything needs to be a batch rn or things break / need rewriting
        # TODO: Decide if it's worth rewriting or just letting it be janky

        # TODO: Fix the need for this ridiculous hotfix
        # Somewhere in input_handler.py
        while len(prompt) > 0 and prompt[0] == self.xft['<s>']:
            prompt = prompt[1:]

        if len(prompt) < self.context_len:
            prompt = [self.xft['<s>']] + prompt
        prompt = prompt[:self.context_len]
        token_batch = torch.tensor([[self.xft['<>']]*(self.context_len - len(prompt)) + prompt])
        #token_batch = torch.tensor([[self.xft['<>']]*(self.context_len - 1) + [self.xft['<s>']]])
        generated_length = 0
        #batch_to_str(token_batch, display=True)
        while generated_length <= response_length:
            #logits, _ = self.forward(token_batch) # (batch_size, seq_len, vocab_size)
            #token_batch = sample(F.softmax(logits, dim = -1)) # batch_size, seq_len)
            # print(f"Returned shape: {token_batch.shape}")
            next = next_token(token_batch)
            if next == self.xft['</s>']:
                break;
            token_batch = torch.cat((token_batch, next), dim=1)
            #batch_to_str(token_batch, display=True)
            generated_length += 1
        output = batch_to_str(token_batch)
        return output
