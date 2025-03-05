from model_handler import train_model
import datanew

model_name = 'abe'
dataset_id = 1741140596
batch_size = 16
context_len = 8
embedding_depth = 8
num_layers = 6
total_heads = 2
masked = True

model = train_model(model_name,
                dataset_id,
                batch_size,
                context_len,
                embedding_depth,
                num_layers,
                total_heads,
                masked)



