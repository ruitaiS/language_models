# requirements.txt add matplotlib
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_progress(df):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.plot(df["batch"], df["loss"], linestyle="-", lw=1, color="blue", label="Loss")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Time Elapsed (s)", color="red")
    ax2.plot(df["batch"], df["time_elapsed"], linestyle="--", lw=1, color="red", label="Time Elapsed")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Loss and Time Elapsed Over Batches")
    ax1.grid()

    plt.show()

df = pd.read_csv("model-0_tr.txt", names=['batch', 'loss', 'time_elapsed'], header=None, sep=' ')  # Load CSV file
print(f"df[:10]: {df[:10]}")
plot_training_progress(df)
