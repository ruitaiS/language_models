import os
import sys
import json
import argparse
import torch
import onnx
from rnn import load_rnn_model

# Unpack hidden state to individual tensors instead of tuple
class RnnWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, h, c):
        logits, (h, c) = self.base_model(x, (h, c))
        return logits, h, c

#------------

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
    model = RnnWrapper(model)
except FileNotFoundError as e:
        print(f"models/{model_version}/{model_filename} not found.")
        print(f"Available in {model_version}:")
        print(files)
        sys.exit(1)

print(f"Loaded model {model_version} epoch {model_epoch}")
print(f"\nModel: {model}")

# Single Token Trace Input:
batch_size, seq_len = (1,1)
example_x = torch.randint(0, model.base_model.vocab_size, (batch_size, seq_len), dtype=torch.long)
example_h, example_c = model.base_model.init_hidden(batch_size)

export_folder = os.path.join('onnx_exports', f'{model_version}_epoch_{model_epoch}')
os.makedirs(export_folder, exist_ok=True)
model.eval()
torch.onnx.export(
        model,
        args=(example_x, example_h, example_c),
        f=os.path.join(export_folder, f'model.onnx'),
        input_names=['x', 'h', 'c'],
        output_names=['logits', 'new_h', 'new_c'],
        dynamic_axes=None,
        dynamo=True
        )

combine=True
if combine:
    split_model = onnx.load(os.path.join(export_folder, f'model.onnx'))
    onnx.save(split_model, os.path.join(export_folder, 'combined.onnx'), save_as_external_data=False)
    os.remove(os.path.join(export_folder, f'model.onnx'))
    os.remove(os.path.join(export_folder, f'model.onnx.data'))
    os.rename(os.path.join(export_folder, 'combined.onnx'), os.path.join(export_folder, 'model.onnx'))

model_assets = {
        'specials': model.base_model.idx2token[:3],
        'chars': "".join(model.base_model.idx2token[3:]),
        'lstm_layers': model.base_model.lstm_layers,
        'hidden_dim': model.base_model.hidden_dim
        }

minify = True
with open(os.path.join(export_folder, 'model_assets.json'), "w", encoding="utf-8") as f:
    if minify:
        json.dump(model_assets, f, ensure_ascii=False, separators=(",", ":"))
    else:
        json.dump(model_assets, f, ensure_ascii=False, indent=2)
