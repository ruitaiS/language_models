# requirements.txt add matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))

def plot_training_progress(df):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.plot(df["batch"], df["loss"], color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Time Elapsed (s)", color="red")
    ax2.plot(df["batch"], df["time_elapsed"], color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Loss, Time vs. Batch")
    ax1.grid()

    plt.show()

df = pd.read_csv(os.path.join(base_path, 'model-01_tr.txt'), names=['batch', 'loss', 'time_elapsed'], header=None, sep=' ')
plot_training_progress(df)
