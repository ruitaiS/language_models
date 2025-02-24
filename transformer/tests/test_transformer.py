import pytest
import torch
import torch.nn as nn
from modules import TransformerBlock, EmbeddingLayer

import random

d = 32
total_heads = 4
vocab_size = 10000
block_size = 64

max_batches = 20

@pytest.fixture
def embedding_layer():
    return EmbeddingLayer(d, vocab_size, block_size)

@pytest.fixture
def transformer_block():
    return TransformerBlock(d, total_heads)

def test_output_shape(transformer_block, embedding_layer):
    batch_size = random.randint(1, max_batches)
    seq_len = random.randint(1, block_size-1)
    print(f'batch_size: {batch_size}, seq_len: {seq_len}, embedding depth: {d}')
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    embeddings, padding_mask = embedding_layer(tokens)
    output = transformer_block(embeddings, padding_mask)

    assert output.shape == (batch_size, seq_len, d), f"Expected ({batch_size}, {seq_len}, {d}), got {output.shape}"

# TODO: Tests for correct math