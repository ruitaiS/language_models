import os
import sys
import argparse
from rnn import load_rnn_model, sample

parser = argparse.ArgumentParser(description="Generate Text from RNN Model")
parser.add_argument('-v', '--version', type=int, required=False, help='Model Version')
parser.add_argument('-e', '--epoch', type=int, required=False, help='Epoch Number')
parser.add_argument('-b', '--book', type=str, required=False, help='Book Name')
args = parser.parse_args()

model_version = 'v' + str(args.version) if args.version is not None else 'v7'
files = [f for f in os.listdir(os.path.join('models', model_version))
         if f.startswith("epoch_") and f.endswith(".net")]
if not files:
    raise FileNotFoundError(f"No model files found in {model_version}")

latest_epoch = int(max(files, key=lambda f: int(f[6:-4]))[6:-4])
model_epoch = int(args.epoch) if args.epoch is not None else latest_epoch
model_filename = f"epoch_{model_epoch}.net"

#---------------------------------------------------------------------------

filepath = os.path.join('models',  model_version, model_filename)
try:
    model, optimizer = load_rnn_model(filepath)
except FileNotFoundError as e:
        print(f"models/{model_version}/{model_filename} not found.")
        print(f"Available in {model_version}:")
        print(files)
        sys.exit(1)

print(f"Loaded model {model_version} epoch {int(model_filename[6:-4])}")
print(f"\nModel: {model}")

'''
# for include_book == true models
# probably unnecessarily complex for now just update manually
params = [f for f in os.listdir(os.path.join('models', model_version))
         if f.startswith("epochs_") and f.endswith(".params")]
'''
if model.tokenization == 'char':
    prime = '\t'
    stop_token='\n'
else:
    prime = '<tab>'
    stop_token = '</s>'

#if model.style == 'encoded_lines':
#    stop_token = model.pad_token

book_included_models = ['v8', 'v9']
if model_version in book_included_models and args.book is not None:
    print(f"Book: {args.book}")
    if model.tokenization == 'char':
        prime = f'<{str(args.book)}>\t'
    else:
        prime = f'<s> <{str(args.book)}> <tab>'

print(f"Tokenization: {model.tokenization}")

text = sample(model, stop_token=stop_token, prime=prime, temperature=1.0)
print(f"\n{text}")
text = sample(model, stop_token=stop_token, prime=prime, temperature=1.0)
print(f"\n{text}")
text = sample(model, stop_token=stop_token, prime=prime, temperature=1.0)
print(f"\n{text}")
