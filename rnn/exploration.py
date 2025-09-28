import matplotlib.pyplot as plt
import numpy as np
from utils import preprocess_akjv

def verse_length_histogram(df, include_book=True,
                    #batch_size, seq_len,
                    #validation_p,
                    shuffle=False):
    #print(f"Batch Size: {batch_size}")
    #print(f"Sequence Length: {seq_len}")
    #print(f"Validation Proportion: {validation_p}")
    print(f"Encoded Verses: {len(df)}")

    verse_lengths = df["text"].str.len()
    bin_size = 5

    bins = np.arange(verse_lengths.min(), verse_lengths.max() + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(verse_lengths, bins=bins, edgecolor="black")
    plt.axvline(verse_lengths.median(), color="green", linestyle="--", linewidth=2, label=f"Median = {verse_lengths.median():.1f}")
    plt.xlabel("Verse Length (characters)")
    plt.ylabel("Frequency")
    plt.show()
    return True


df, _ = preprocess_akjv()
verse_length_histogram(df)
