import os
import torch
import json

base_path = os.path.dirname(os.path.abspath(__file__))
def save(model, name):
    torch.save(model.state_dict(), os.path.join(base_path, f"models/{name}.pth"))
    with open(os.path.join(base_path, f"models/{name}_params.json"), 'w') as f:
        json.dump(model.params(), f, indent=4)