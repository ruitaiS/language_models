import pytest
import torch
import torch.nn as nn
from main import EmbeddingLayer

import random

EMBED_DIM = 32
VOCAB_SIZE = 10000
BLOCK_SIZE = 64

MAX_BATCHES = 20

@pytest.fixture
def embedding_layer():
    return EmbeddingLayer(EMBED_DIM, VOCAB_SIZE, BLOCK_SIZE)


def test_output_shape(embedding_layer):
    batch_size = random.randint(1, MAX_BATCHES)
    seq_len = random.randint(1, BLOCK_SIZE-1)
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    output = embedding_layer(tokens)

    assert output.shape == (batch_size, seq_len, EMBED_DIM), f"Expected ({batch_size}, {seq_len}, {EMBED_DIM}), got {output.shape}"

def test_token_embedding_shape(embedding_layer):
    batch_size = random.randint(1, MAX_BATCHES)
    seq_len = random.randint(1, BLOCK_SIZE-1)
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    token_embeddings = embedding_layer.E(tokens)

    assert token_embeddings.shape == (batch_size, seq_len, EMBED_DIM), \
        f"Expected ({batch_size}, {seq_len}, {EMBED_DIM}), got {token_embeddings.shape}"
    
def test_position_embedding_shape(embedding_layer):
    batch_size = random.randint(1, MAX_BATCHES)
    seq_len = random.randint(1, BLOCK_SIZE-1)
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    pos_embeddings = embedding_layer.P(positions)

    assert pos_embeddings.shape == (batch_size, seq_len, EMBED_DIM), \
        f"Expected ({batch_size}, {seq_len}, {EMBED_DIM}), got {pos_embeddings.shape}"

def test_composite_embedding_sum(embedding_layer):
    """Test that the sum of token and position embeddings produces valid embeddings."""
    batch_size = random.randint(1, MAX_BATCHES)
    seq_len = random.randint(1, BLOCK_SIZE-1)
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

    output = embedding_layer(tokens)
    token_embeddings = embedding_layer.E(tokens)
    
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    pos_embeddings = embedding_layer.P(positions)

    expected_output = token_embeddings + pos_embeddings

    assert torch.allclose(output, expected_output, atol=1e-5), \
        "Composite embedding sum does not match expected output."