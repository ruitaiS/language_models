import os
import json
import torch
from torch.nn import functional as F

base_path = os.path.dirname(os.path.abspath(__file__))
def save(model, name):
    torch.save(model.state_dict(), os.path.join(base_path, f"models/{name}.pth"))
    with open(os.path.join(base_path, f"models/{name}_params.json"), 'w') as f:
        json.dump(model.params(), f, indent=4)

def generate(model, tokenizer, prompt=[], max_length=500):
    # TODO: temperature, top-k (see ch 10)
    model.eval()
    tokens = torch.tensor([[tokenizer.start_token_idx] + prompt], device=model.device())
    for _ in range(max_length):
        logits = model(tokens)[:, -1:, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.distributions.Categorical(probs=probs).sample()
        tokens = torch.cat([tokens, next_token], dim=-1)
        if next_token == tokenizer.end_token_idx:
            break
    print(f"{tokenizer.decode(tokens[0].tolist())}\n")
    model.train()