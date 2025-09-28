import os
import sys
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) == 2:
    user_input = sys.argv[1]
else:
    user_input = 'v6'
path = os.path.join('models', user_input, 'training_metadata')

df_rows = []
for filename in os.listdir(path):
    if filename.startswith("epoch_") and filename.endswith(".meta"):
        match = re.match(r"epoch_(\d+)\.meta", filename)
        if not match:
            continue
        epoch = int(match.group(1))

        with open(os.path.join(path, filename), "r") as f:
            data = json.load(f)

        epoch_losses = data['epoch_losses']
        val_loss = data.get('val_loss', 0)

        for loss in epoch_losses:
            df_rows.append({
                "epoch": epoch,
                "loss": loss,
                "val_loss": val_loss,
                })

if len(df_rows) == 0:
    print(f"No metadata found in models/{user_input}/training_metadata")
else:

    df = pd.DataFrame(df_rows)
    df = df.sort_values(['epoch'], kind="stable").reset_index(drop=True)
    df['total_batches'] = range(1, len(df)+1)

    #-----------------

    plt.figure(figsize=(12, 6))
    plt.plot(df['total_batches'], df['loss'], label="Training Loss")
    plt.plot(df['total_batches'], df['val_loss'], label="Validation Loss")

    epoch_bounds = df.groupby("epoch")["total_batches"].min()
    for epoch, bound in epoch_bounds.items():
        plt.axvline(x=bound, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(epoch_bounds, labels=epoch_bounds.index)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"RNN Model {user_input}")
    plt.legend()
    plt.tight_layout()
    plt.show()

