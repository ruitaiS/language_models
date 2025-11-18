import torch
from torch.nn import functional as F
from data import Tokenizer
#import transformer

def decode(encoded_tensor):
    return [idx2token(token.item()) for token in encoded_tensor]

# TODO (this is directly cut out of the transformer module + needs edits)
def generate(model, idx2token, prompt= [], response_length=100):
    def sample(probabilities):
        # TODO See ch 10; top-k should be easy
        #print(f"Token Probabilities Shape: {probabilities.shape}")
        return torch.distributions.Categorical(probs=probabilities).sample()

        #return torch.argmax(probabilities, dim =-1)
    def next_token(token_batch):
        # longer sequences (eg. those produced during sampling) truncated to maximum context length
        token_batch = token_batch[:, -model.context_len:]
        logits, _ = model.forward(token_batch)
        logits = logits[:, -1:, :] # Only want last token, not whole sequence
        #print(f"Logits shape: {logits.shape}")
        # dim=-1 >> softmax along vocab indices to get probabilities
        probabilities = F.softmax(logits, dim=-1)
        #print(f'lm probabilities: {probabilities}')
        return sample(probabilities)

    def batch_to_str(token_batch, display=False):
        output = [idx2token[token.item()] for token in token_batch.squeeze(0)]
        if display: print(' '.join(output))
        return output

    # starting batch with batch_size = 1, seq_len = 1
    # Everything needs to be a batch rn or things break / need rewriting
    # TODO: Decide if it's worth rewriting or just letting it be janky

    # TODO: Fix the need for this ridiculous hotfix
    # Somewhere in input_handler.py
    while len(prompt) > 0 and prompt[0] == start_token_idx:
        prompt = prompt[1:]

    if len(prompt) < model.context_len:
        prompt = [start_token_idx] + prompt
    prompt = prompt[:model.context_len]
    token_batch = torch.tensor([[pad_token_idx]*(model.context_len - len(prompt)) + prompt])
    #token_batch = torch.tensor([[pad_token_idx]*(model.context_len - 1) + [start_token_idx]])
    generated_length = 0
    #batch_to_str(token_batch, display=True)
    while generated_length <= response_length:
        #logits, _ = model.forward(token_batch) # (batch_size, seq_len, vocab_size)
        #token_batch = sample(F.softmax(logits, dim = -1)) # batch_size, seq_len)
        # print(f"Returned shape: {token_batch.shape}")
        next = next_token(token_batch)
        if next == end_token_idx:
            break;
        token_batch = torch.cat((token_batch, next), dim=1)
        #batch_to_str(token_batch, display=True)
        generated_length += 1
    output = batch_to_str(token_batch)
    return output