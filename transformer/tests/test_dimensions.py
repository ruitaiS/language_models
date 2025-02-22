import torch
import pytest
from modules import SHA, MHA, FFN, LayerNorm, TransformerBlock, LanguageModelHead, LanguageModel
import data

masked = True
batch_size = 2
seq_len = 4


embed_dim = 8 # d
total_heads = 2
head_dim = embed_dim / total_heads
num_layers = 6

xft, tfx = data.get_vocab()
vocab_size = len(xft)
print(f'Vocab Size: {vocab_size}')

input_tokens, target_tokens = data.sample(batch_size, seq_len)

X = torch.randn(batch_size, seq_len, embed_dim)
#---------------------------

@pytest.fixture
def multi_attention(): 
    return MHA(d = embed_dim, total_heads = total_heads, masked=masked)

@pytest.fixture
def ffn(): 
    return FFN(d = embed_dim)

@pytest.fixture
def layernorm(): 
    return LayerNorm(d = embed_dim)

@pytest.fixture
def transformerblock(): 
    return TransformerBlock(d = embed_dim, total_heads = total_heads)

@pytest.fixture
def lm_head(language_model): 
    return language_model.lm_head

@pytest.fixture
def language_model(): 
    return LanguageModel(embed_dim, data.get_vocab(), seq_len, num_layers, total_heads)

def test_multi_attention_output_shape(multi_attention):
    output = multi_attention(X)
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_ffn_output_shape(ffn):
    output = ffn(X)
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_layernorm_output_shape(layernorm):
    output = layernorm(X)
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_block_output_shape(transformerblock):
    output = transformerblock(X)
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_lm_output_shape(language_model):
    logits, _ = language_model(input_tokens)
    logits_expected_shape = (batch_size, seq_len, vocab_size)
    
    assert logits.shape == logits_expected_shape, f"Expected {logits_expected_shape}, got {logits.shape}"