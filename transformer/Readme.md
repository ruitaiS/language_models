### transformer language model

#### Run the model (make sure venv is active):
```
python main.py
```

Right now, the model runs autoregressive text continuation based on corrected user input.

There are two methods for input correction, direct lookup, which tries to find an exact match for the words in the corpus vocabulary, and Hidden Markov Model based correction. The code for the HMM method is based on the sentence correction portion of the `ecse_526` assignment. You can toggle between these methods by setting the  `correction_method` variable to `1` or `2` on `line 96` in the file `main.py`

After correcting the user input, the model will run text continuation using the input as the initial seed phrase. This loops until the user types `quit` or `q!`.

___

#### notes

The transformer implementation can be found in `modules.py`.

I tried as much as possible to follow the schematic outlined in the 2017 *Attention is All You Need* paper. I also drew from several chapters of Jurafsky and Martin's *Speech and Language Processing* and watched a lot of YouTube to fill in the gaps. I've forgotten the links to the YouTube videos but the relevant PDFs can be found in the `pdf` subfolder.

So far this has mainly been an exercise in building out and debugging the architecture. as it stands the output isn't great, but everything should be working together properly (as far as I can tell). Tuning the output is the next step, but truthfully I don't have a very good understanding of how to tune yet, so I'll have to revisit this after I go down that rabbit hole.
