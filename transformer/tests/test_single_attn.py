import torch
import pytest
from main import SHA

use_mask = True
batch_size = 2
seq_len = 4
embed_dim = 8 # d
head_dim = 4  # (aka d_k aka d_v)

# TODO: Tests for Explicitly Single Attention Mode
# Different d_k and d_v
# Internal W_0 and outputting shape (batch_size, seq_len, d)

# As part of Multi-Headed Attention: ---------------------------------------------------------------
# d_k and d_v are both set to head_dim

# The multi-headed attention module will aggregate the outputs from each single attention head
# and apply it's own W_0 to all of them, converting back into vectors of length d from head_dim
# for the final output with shape (batch_size, seq_len, d)

#-----------------------------------------

# All inputs are this format:
# X.shape = (batch_size, seq_len, d)

# Every single attention head as part of a multi-head module should output
# shape (batch_size, seq_len, head_dim)

@pytest.fixture
def single_attention(): 
    return SHA(d = embed_dim, d_k = head_dim, d_v = head_dim, use_mask=use_mask)

def test_final_output(single_attention):
    X = torch.randn(batch_size, seq_len, embed_dim)
    Q = single_attention.W_Q(X)
    K = single_attention.W_K(X)
    V = single_attention.W_V(X)

    scores = single_attention.scaled_dot_prod(Q, K)
    if use_mask: scores = single_attention.mask(scores)
    weights = torch.softmax(scores, dim=-1)

    output = single_attention(X)

    # Check output is correct shape
    expected_shape = (batch_size, seq_len, head_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # Check multiplication is correct
    torch.testing.assert_close(output, weights @ V)

def test_linear_transforms(single_attention):
    X = torch.randn(batch_size, seq_len, embed_dim)
    Q = single_attention.W_Q(X)
    K = single_attention.W_K(X)
    V = single_attention.W_V(X)

    # Check outputs are correct shape
    expected_shape = (batch_size, seq_len, head_dim)
    assert Q.shape == expected_shape, f"Expected {expected_shape}, got {Q.shape}"
    assert K.shape == expected_shape, f"Expected {expected_shape}, got {K.shape}"
    assert V.shape == expected_shape, f"Expected {expected_shape}, got {V.shape}"

    # TODO: Check values change during learning

def test_masking(single_attention):
    scores = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
    masked_scores = single_attention.mask(scores)

    for i in range(batch_size): # i indexes sequences in the batch
        for j in range(seq_len): # j is the index for the attention vector for the jth token in the sequence
            for k in range(seq_len): # k is the index on the attention vector that contains the attention score from the jth token to the kth token in the sequence
                # Think of j as the current token, and k as the one you're looking at
                # If k is behind j, it should be visible, eg it should match the unmasked value
                # if k is ahead of j, it should be masked out
                if (j >= k) or (not use_mask):
                    assert masked_scores[i, j, k] == scores[i, j, k], f"Value mismatch:\
                    masked[{i},{j},{k}] = {masked_scores[i, j, k]}, unmasked[{i},{j},{k}] = {scores[i, j, k]}"
                else:
                    assert masked_scores[i, j, k] == -float('inf'), f"Expected masked[{i},{j},{k}], got {masked_scores[i, j, k]}"

def test_attention_scores(single_attention):
    X = torch.randn(batch_size, seq_len, embed_dim)
    Q = single_attention.W_Q(X)
    K = single_attention.W_K(X)
    
    # Check dot prod math is correct
    scores = single_attention.scaled_dot_prod(Q, K)
    torch.testing.assert_close(scores, Q @ K.transpose(-2, -1) / (head_dim ** 0.5))

    # Check masking is working
    if use_mask:
        scores = single_attention.mask(scores)
        # TODO: Test Mask Values

    # Check output is correct shape
    assert scores.shape == (batch_size, seq_len, seq_len), \
        f"Expected attention score shape {(batch_size, seq_len, seq_len)}, got {scores.shape}"

def test_weighted_sums(single_attention):
    X = torch.randn(batch_size, seq_len, embed_dim)
    Q = single_attention.W_Q(X)
    K = single_attention.W_K(X)
    V = single_attention.W_V(X)

    scores = single_attention.scaled_dot_prod(Q, K)
    if use_mask: scores = single_attention.mask(scores)

    weights = torch.softmax(scores, dim=-1)

    # Check weights sum to 1
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(batch_size, seq_len))

def test_gradient_flow(single_attention):
    X = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    output = single_attention(X).sum()
    output.backward()  # Compute gradients

    # Check there are parameters
    assert len(list(single_attention.parameters())) > 0, "No parameters"

    # Check the parameters are assigned gradients
    for param in single_attention.parameters():
        assert param.grad is not None, f"Gradient not found for {param}"

def test_reproducibility():
    """Ensure model produces the same output when given the same seed."""
    torch.manual_seed(42)
    model1 = SHA(embed_dim, head_dim, head_dim)
    
    torch.manual_seed(42)
    model2 = SHA(embed_dim, head_dim, head_dim)

    x = torch.randn(batch_size, seq_len, embed_dim)
    output1 = model1(x)
    output2 = model2(x)

    torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6)
