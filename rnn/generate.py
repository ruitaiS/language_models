import os
import sys
from rnn import load_rnn_model, sample

if len(sys.argv) > 1:
    model_version = sys.argv[1]
else:
    model_version = 'v6'

files = [f for f in os.listdir(os.path.join('models', model_version))
         if f.startswith("epoch_") and f.endswith(".net")]
if not files:
    raise FileNotFoundError(f"No model files found in {model_version}")

latest_epoch = int(max(files, key=lambda f: int(f[6:-4]))[6:-4])
model_filename = f"epoch_{latest_epoch}.net"
if len(sys.argv) > 2:
    model_filename = f"epoch_{int(sys.argv[2])}.net"

#---------------------------------------------------------------------------

filepath = os.path.join('models',  model_version, model_filename)
try:
    model, optimizer = load_rnn_model(filepath)
except FileNotFoundError as e:
        print(f"models/{model_version}/{model_filename} not found.")
        print(f"Available in {model_version}:")
        print(files)
        sys.exit(1)

print(f"Loading model {model_version} epoch {int(model_filename[6:-4])}")
print(f"\nModel: {model}")

text = sample(model, stop_char='\n', prime='\t', temperature=1.0)
print(f"\n{text}")
text = sample(model, stop_char='\n', prime='\t', temperature=1.0)
print(f"\n{text}")
text = sample(model, stop_char='\n', prime='\t', temperature=1.0)
print(f"\n{text}")
