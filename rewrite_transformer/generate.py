import torch
from torch.nn import functional as F

# TODO: temperature, top-k (see ch 10)
# Samples the last logits for each element in the batch
def sample(logits_batch):
    logits_batch = logits_batch[:, -1:, :]
    probs = F.softmax(logits_batch, dim=-1)
    return torch.distributions.Categorical(probs=probs).sample()

def test_generate(model, tokenizer, prompt=[], max_length=500):
    model.eval()
    tokens = torch.tensor([[tokenizer.start_token_idx] + prompt], device=model.device())
    for _ in range(max_length):
        logits = model(tokens)
        next_token = sample(logits)
        tokens = torch.cat([tokens, next_token], dim=-1)
        if next_token == tokenizer.end_token_idx:
            break
    print(f"{tokenizer.decode(tokens[0].tolist())}\n")
    model.train()