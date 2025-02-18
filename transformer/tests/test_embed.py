import pytest
import torch
import torch.nn as nn
from main import EmbeddingLayer

import random

embed_dim = 32
vocab_size = 10000
block_size = 64

max_batches = 20

@pytest.fixture
def embedding_layer():
    return EmbeddingLayer(embed_dim, vocab_size, block_size)

def test_embeddings(embedding_layer):
    batch_size = random.randint(1, max_batches)
    seq_len = random.randint(1, block_size-1)

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
    seq_len = random.randint(1, block_size-1)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    token_embeddings = embedding_layer.E(tokens)
    pos_embeddings = embedding_layer.P(positions)

    expected_output = token_embeddings + pos_embeddings
    output = embedding_layer(tokens)

    # Check shape
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected ({batch_size}, {seq_len}, {embed_dim}), got {output.shape}"

    # Check sum is correct
    torch.testing.assert_close(output, expected_output)