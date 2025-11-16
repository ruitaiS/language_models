import data
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from transformer import LanguageModel

def train_model(model_name,
                batch_size,
                context_len,
                embedding_depth,
                num_layers,
                total_heads,
                masked = True):

    # TODO
    #dataset = data.get_dataset('train', dataset_id)
    #aux_data = data.extract_aux(dataset)

    idx2token, token2idx = None # TODO

    model = LanguageModel(
            (idx2token, token2idx),
            embedding_depth,
            context_len,
            num_layers,
            total_heads)
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
        # Training Loss Info
        #with open(os.path.join(base_path, f'models/{model_name}_tr.txt'), 'w') as f:
        #    writer = csv.writer(f, delimiter=' ')

            #start = time.time()
            for batch_index, (X, Y) in enumerate(zip(input_batches, target_batches)):
                logits, loss = model(X, targets=Y)
                optimizer.zero_grad(set_to_none=True)
                #optimizer.grad.zero_()
                loss.backward()
                #print("Optimizer grad: ", optimizer.grad)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                #elapsed = time.time() - start
                #writer.writerow([batch_index, loss.item(), elapsed])
                if batch_index % print_interval == 1:
                    #seconds = (elapsed / batch_index)*(total_batches - batch_index)
                    #minutes = int(seconds/60)
                    #seconds = int(seconds % 60)
                    #print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}. Estimated time remaining: {minutes}m{seconds}s")
                    print(f"Batch {batch_index} of {total_batches}. Loss: {loss:.2f}.")
    except KeyboardInterrupt:
        #f.close()
        torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
        with open(os.path.join(base_path, f"models/{model_name}_params.json"), 'w') as f:
            json.dump(model_params, f, indent=4)

    f.close()
    torch.save(model.state_dict(), os.path.join(base_path, f"models/{model_name}.pth"))
    with open(os.path.join(base_path, f"models/{model_name}_params.json"), 'w') as f:
        json.dump(model_params, f, indent=4)

    return model



'''train_model(model_name = 'trigram_imitation'
dataset_id = 1741140596
batch_size = 16
context_len = 3
embedding_depth = 8
num_layers = 6
total_heads = 1
masked = True)'''

tokenization='char'
include_book=True
if tokenization=='char':
    delimiter = ''
else:
    delimiter = ' '

processed_lines= data.preprocess_akjv(include_book)
encoded_lines, vocab_size, idx2token, token2idx = data.build_and_encode(processed_lines, tokenization)

print(f"Sample Line Encoded:\n{encoded_lines[0]}\n")
print(f"Sample Line Reconstructed:\n{delimiter.join([idx2token.get(idx, '<?>') for idx in encoded_lines[0]])}\n")