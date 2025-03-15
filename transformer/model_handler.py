import os
import time
import csv
import json
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

import data
from modules import LanguageModel
base_path = os.path.dirname(os.path.abspath(__file__))

def train_model(model_name,
                dataset_id,
                batch_size, # might be irrelevant for the model
                context_len,
                embedding_depth,
                num_layers,
                total_heads,
                masked = True):

    model_params = {'model_name': model_name,
                    'dataset_id': dataset_id,
                    'batch_size': batch_size,
                    'context_len': context_len,
                    'embedding_depth': embedding_depth,
                    'num_layers': num_layers,
                    'total_heads': total_heads,
                    'masked': True
            }

    dataset = data.get_dataset('train', dataset_id)
    aux_data = data.extract_aux(dataset)

    model = LanguageModel(
            aux_data['vocab'],
            model_params['embedding_depth'],
            model_params['context_len'],
            model_params['num_layers'],
            model_params['total_heads'])
    model.train()

    print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
    for p in model.parameters():
        p.requires_grad_(True)
        #p.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    print(f"Optimizer: {optimizer}")

    input_batches, target_batches = data.batch(batch_size, context_len, dataset)
    total_batches = len(input_batches)
    print_interval = 500

    try:
        with open(os.path.join(base_path, f'models/{model_name}_tr.txt'), 'w') as f:
            writer = csv.writer(f, delimiter=' ')

            start = time.time()
            for batch_index, (X, Y) in enumerate(zip(input_batches, target_batches)):
                logits, loss = model(X, targets=Y)
                optimizer.zero_grad(set_to_none=True)
                #optimizer.grad.zero_()
                loss.backward()
                #print("Optimizer grad: ", optimizer.grad)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                elapsed = time.time() - start
                writer.writerow([batch_index, loss.item(), elapsed])
                if batch_index % print_interval == 1:
                    seconds = (elapsed / batch_index)*(total_batches - batch_index)
                    minutes = int(seconds/60)
                    seconds = int(seconds % 60)
                    print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}. Estimated time remaining: {minutes}m{seconds}s")
    except KeyboardInterrupt:
        f.close()
        torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
        with open(os.path.join(base_path, f"models/{model_name}_params.json"), 'w') as f:
            json.dump(model_params, f, indent=4)

    f.close()
    torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
    with open(os.path.join(base_path, f"models/{model_name}_params.json"), 'w') as f:
        json.dump(model_params, f, indent=4)

    return model

def load_model(model_name, mode='eval'):
    meta_fp = os.path.join(base_path, f'models/{model_name}_params.json')
    model_fp = os.path.join(base_path, f'models/{model_name}.pth')
    model_params = {}
    try:
        with open(meta_fp, 'r') as f:
            model_params = json.load(f)
    except Exception as e:
        print(e)
        print(f'Error loading metadata for model {model_name}.')
        return None
    dataset = data.get_dataset('train', model_params['dataset_id'])
    aux_data = data.extract_aux(dataset)

    model = LanguageModel(
            aux_data['vocab'],
            model_params['embedding_depth'],
            model_params['context_len'],
            model_params['num_layers'],
            model_params['total_heads'])
    model.load_state_dict(torch.load(model_fp))

    modes = {
        'eval' : model.eval,
        'train': model.train,
    }
    assert mode in modes.keys(), f'Invalid operation mode: {mode}'
    modes[mode]()
    #model.eval()

    print(f"Model: {model_name}.pth")
    print(f"Vocabulary Size: {len(aux_data['vocab'][0])}\nContext Length: {model_params['context_len']}")
    print("Embedding Depth: ", model_params['embedding_depth'])
    print(f"Layers: {model_params['num_layers']}\nHeads: {model_params['total_heads']}")
    print(f"Total Parameters: {sum(p.nelement() for p in model.parameters())}")
    print(f"Operation Mode: {'train' if model.training else 'eval'}")

    return {
        'core': model,
        'params': model_params,
        'vocab': aux_data['vocab'],
        'bigram': aux_data['bigram']
    }
