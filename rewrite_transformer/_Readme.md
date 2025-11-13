TODOs:

tokenizer concerns (tokenization schema, idx2token and token2idx) should be kept seperate from the model

sampling / generation logic should also be seperated from the model. model should keep `forward` function, which is called by sampling code in conjunction with tokenizer

model should only ever interact with encoded token indices; the tokenizer class should handle encoding/decoding

model really only needs to know vocab size, and indices for special tokens, eg. padding and start/end tokens

training / loss concerns should also be it's own module; should not be packaged in the model. model should only have a forward function >> should NOT contain optimizer / loss functions

