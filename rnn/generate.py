import os
from rnn import load_rnn_model, sample

filepath = os.path.join('models',  'v6', 'epoch_50.net')
model, optimizer = load_rnn_model(filepath)
print(f"\nModel: {model}")

text = sample(model, stop_char='\n', prime='\t', temperature=1.0)
print(f"\n{text}")
text = sample(model, stop_char='\n', prime='\t', temperature=1.0)
print(f"\n{text}")
text = sample(model, stop_char='\n', prime='\t', temperature=1.0)
print(f"\n{text}")
