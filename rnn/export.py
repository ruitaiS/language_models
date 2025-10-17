import os
import sys
import json
import argparse
import torch
from rnn import load_rnn_model

parser = argparse.ArgumentParser(description="Select RNN Model For Export")
parser.add_argument('-v', '--version', type=int, required=False, help='Model Version')
parser.add_argument('-e', '--epoch', type=int, required=False, help='Epoch Number')
args = parser.parse_args()

model_version = 'v' + str(args.version) if args.version is not None else 'v11'
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

print(f"Loaded model {model_version} epoch {model_epoch}")
print(f"\nModel: {model}")

# Single Token Trace Input:
batch_size, seq_len = (1,1)
example_x = torch.randint(0, model.vocab_size, (batch_size, seq_len), dtype=torch.long)
example_hidden = model.init_hidden(batch_size)
h, c = example_hidden

os.makedirs('onnx_exports', exist_ok=True)
model.eval()
torch.onnx.export(
        model,
        args=(example_x, example_hidden),
        f=os.path.join('onnx_exports', f'{model_version}_epoch_{model_epoch}.onnx'),
        input_names=['x', 'hidden'],
        output_names=['logits', 'new_hidden'],
        dynamic_axes=None,
        dynamo=True
        )

model_assets = {
        'specials': model.idx2token[:3],
        'chars': "".join(model.idx2token[3:]),
        'lstm_layers': model.lstm_layers,
        'hidden_dim': model.hidden_dim
        }

with open(os.path.join('onnx_exports', f'{model_version}_epoch_{model_epoch}_assets.json'), "w", encoding="utf-8") as f:
    json.dump(model_assets, f, ensure_ascii=False, indent=2)
