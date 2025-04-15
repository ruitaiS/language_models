### transformer language model

#### Run the model:
```
python main.py
```

As-is, the model runs autoregressive text continuation based on corrected user input.

There are two methods for input correction, direct lookup, which tries to find an exact match for the words in the corpus vocabulary, and Hidden Markov Model based correction. The code for the HMM method is based on the sentence correction portion of the `ecse_526` assignment. You can toggle between these methods by setting the  `correction_method` variable to `1` or `2` on `line 96` in the file `main.py`

After correcting the user input, the model will run text continuation using the input as the initial seed phrase. This loops until the user types `quit` or `q!`.
