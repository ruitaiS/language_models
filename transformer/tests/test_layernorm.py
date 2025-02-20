import pytest
import torch
import torch.nn as nn
from modules import EmbeddingLayer, LayerNorm

import random

d = 32
vocab_size = 10000
block_size = 64

max_batches = 20

@pytest.fixture
def embedding_layer():
    return EmbeddingLayer(d, vocab_size, block_size)

@pytest.fixture
def layer_norm():
    return LayerNorm(d)

def test_output_shape(layer_norm, embedding_layer):
    batch_size = random.randint(1, max_batches)
    seq_len = random.randint(1, block_size-1)
    print(f'batch_size: {batch_size}, seq_len: {seq_len}, embedding depth: {d}')
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    embeddings = embedding_layer(tokens)
    output = layer_norm(embeddings)

    assert output.shape == (batch_size, seq_len, d), f"Expected ({batch_size}, {seq_len}, {d}), got {output.shape}"

# TODO: Tests for math