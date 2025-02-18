import torch
import pytest
from main import SHA, MHA, FFN, LayerNorm, TransformerBlock

use_mask = True
batch_size = 2
seq_len = 4
embed_dim = 8 # d
head_dim = 4  # (aka d_k aka d_v)
total_heads = 2

# head_dim on SHA = embed_dim / total_heads on MHA

X = torch.randn(batch_size, seq_len, embed_dim)

@pytest.fixture
def multi_attention(): 
    return MHA(d = embed_dim, total_heads = total_heads, use_mask=use_mask)

@pytest.fixture
def ffn(): 
    return FFN(d = embed_dim)

@pytest.fixture
def layernorm(): 
    return LayerNorm(d = embed_dim)

@pytest.fixture
def transformerblock(): 
    return TransformerBlock(d = embed_dim, total_heads = total_heads)


def test_multi_attention_output_shape(multi_attention):
    output = multi_attention(X)
    # Check output is correct shape
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_ffn_output_shape(ffn):
    output = ffn(X)
    # Check output is correct shape
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_layernorm_output_shape(layernorm):
    output = layernorm(X)
    # Check output is correct shape
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_block_output_shape(transformerblock):
    output = transformerblock(X)
    # Check output is correct shape
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
