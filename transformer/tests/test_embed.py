import pytest
import torch
from modules import EmbeddingLayer, SHA

import random

vocab_size = 10000
context_len = 8

max_batches = 20

masked = True
embed_dim = 8 # d
head_dim = 4  # (aka d_k aka d_v)

@pytest.fixture
def embedding_layer():
    return EmbeddingLayer(embed_dim, vocab_size, context_len)

@pytest.fixture
def single_attention(): 
    return SHA(d = embed_dim, d_k = head_dim, d_v = head_dim, masked=masked)

def test_embeddings(embedding_layer):
    batch_size = random.randint(1, max_batches)
    seq_len = random.randint(1, context_len-1)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    token_embeddings = embedding_layer.E(tokens)
    pos_embeddings = embedding_layer.P(positions)

    # Check token embeddings are correct shape
    assert token_embeddings.shape == (batch_size, seq_len, embed_dim), \
        f"Expected ({batch_size}, {seq_len}, {embed_dim}), got {token_embeddings.shape}"
    
    # Check position embeddings are correct shape
    assert pos_embeddings.shape == (batch_size, seq_len, embed_dim), \
        f"Expected ({batch_size}, {seq_len}, {embed_dim}), got {pos_embeddings.shape}"

def test_final_output(embedding_layer):
    batch_size = random.randint(1, max_batches)
    seq_len = random.randint(1, context_len-1)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    token_embeddings = embedding_layer.E(tokens)
    pos_embeddings = embedding_layer.P(positions)

    expected_output = token_embeddings + pos_embeddings
    output, _ = embedding_layer(tokens)

    # Check shape
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected ({batch_size}, {seq_len}, {embed_dim}), got {output.shape}"

    # Check sum is correct
    torch.testing.assert_close(output, expected_output)

# TODO: This doesn't work the way you think
# The intention is to check for proper layering of autoregressive vs. padding masks
# but this doesn't do that rn lol
def test_padding_mask(embedding_layer, single_attention):
    batch_size = 8
    seq_len = 8

    tokens = torch.tensor([[8, 8, 8, 8, 8, 8, 8, 0],
                           [8, 8, 8, 8, 8, 8, 0, 1], 
                           [8, 8, 8, 8, 8, 0, 1, 2], 
                           [8, 8, 8, 8, 0, 1, 2, 3], 
                           [8, 8, 8, 0, 1, 2, 3, 4], 
                           [8, 8, 0, 1, 2, 3, 4, 5], 
                           [8, 0, 1, 2, 3, 4, 5, 6], 
                           [0, 1, 2, 3, 4, 5, 6, 7],
                           ])
    
    _, padding_mask = embedding_layer(tokens)
    scores = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
    masked_scores = single_attention.mask(scores, padding_mask)
    print(masked_scores)
    assert True