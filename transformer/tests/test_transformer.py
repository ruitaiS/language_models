import pytest
import torch
import torch.nn as nn
from main import TransformerBlock, EmbeddingLayer

import random

EMBED_DIM = 32
VOCAB_SIZE = 10000
BLOCK_SIZE = 64

MAX_BATCHES = 20

@pytest.fixture
def embedding_layer():
    return EmbeddingLayer(EMBED_DIM, VOCAB_SIZE, BLOCK_SIZE)

@pytest.fixture
def transformer_block():
    return TransformerBlock()

def test_output_shape(transformer_block, embedding_layer):
    batch_size = random.randint(1, MAX_BATCHES)
    seq_len = random.randint(1, BLOCK_SIZE-1)
    print(f'batch_size: {batch_size}, seq_len: {seq_len}, embedding depth: {EMBED_DIM}')
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    embeddings = embedding_layer(tokens)
    transformer_output = transformer_block(embeddings)

    assert transformer_output.shape == (batch_size, seq_len, EMBED_DIM), f"Expected ({batch_size}, {seq_len}, {EMBED_DIM}), got {transformer_output.shape}"