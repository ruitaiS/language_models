### responsibility seperation

tokenizer concerns (tokenization schema, idx2token and token2idx) should be kept seperate from the model

sampling / generation logic should also be seperated from the model. model should keep `forward` function, which is called by sampling code in conjunction with tokenizer

model should only ever interact with encoded token indices; the tokenizer class should handle encoding/decoding

model really only needs to know vocab size, and indices for special tokens, eg. padding and start/end tokens

training / loss concerns should also be it's own module; should not be packaged in the model. model should only have a forward function >> should NOT contain optimizer / loss functions

for RNN, `init_hidden` is a legit part of the model >> we want to be able to call it during training to reset the hidden state, and we also want the flexibility to define batch size / device from within the training function, rather than cement it into the model definition

### proper instantiation & robust model handling

to save the model you need to store more than just the model weights themselves
- dataest, dataloaders
- model parameters
- sampling logic needs to share the same vocabulary lookup that the model is using
- optimizer state if you're going to resume training
- training parameters if you want to roll back training to a certain epoch

and even though we just mentioned in the earlier section that they're seperate, all the supporting codebase still all need to read from a shared configuration state

- the Dataset class needs to share a context_len parameter with the model to know what size sequences to return
- the model still needs to know the total vocabulary size, as well as indices for special tokens
- [other things that i can't think of rn]

really worth thinking about how to properly package these so that everything is instantiated together properly. i remember with the RNN, every time i tweaked one of the params it would break all my earlier models b/c now the new vocab format doesn't work, or this or that other dependency broke. however you end up saving / loading these models, you need to make sure it's a reasonably self-contained chunk that can actually load in everything it needs.

you basically need to store the entire state of the surrounding ecosystem around the model

probably worth making a dependency diagram for this too, just so you internally have a working understanding of how everything slots in together and documentation for when you inevitably forget. once you start tweaking stuff, it becomes really hard to remember all the changes and the cascading dependency adjustments, so that process needs to be well documented and as self-contained / idiot-proof as possible

### regarding building the vocab from the full text:

in one of the writeups (i think the rough draft for the bible rnn) i mentioned feeling guilty because i'd used the full text corpus to create the vocab rather than restricting the vocabulary to words in the training set & handling out of vocab words at validation/test time, but it turns out this is actually common practice even in production LLMs; nobody else wants to deal with OoV either. So basically i'm not wrong and everyone agrees with me >:)

from my understanding it's b/c out of vocab words are an unavoidable source of error caused by the distribution of rare words in the dataset, rather than the result of anything that can be addressed through training. vocabulary building and model training are seperate processes; knowing which words exist doesn't actually leak any exploitable information to the model.

i'd compare it to defining ahead of time an exhaustive list of the possible output categories for a classification problem - not doing so would be like training a cat / dog classifier and then dinging the model because there's randomly a fish in the test set, yk?

### efficiently extracting context windows from the dataset

the dumb way (the way i did it before lol) would be to pre-calculate all the possible context window examples that you can make from the dataset, and just store them. the reason it's tremendously dumb is b/c the context windows overlap so you're storing massively and unnecessarily redundant data.

>> the phrase "unnecessarily redundant" is itself unnecessarily redundant :))

assuming each line starts with the start token, and ends with the end token, and we're sliding the context windows with stride length 1, you get `len(line) - 1` possible context window `x, y` pairs from each line.

say each line is `[<s>, t1, t2, ... tn, </s>]`
the last element of x would slide from `<s>` to `tn`, and the last element of y would slide from `t1` to `</s>`. the rest of the context window is filled from preceeding elements from the line or left-padded with the pad token.

>> `stride > 1` complicates the math, makes line endings messier since line length isn't guaranteed to be evenly divisible by stride, meaning you'll frequently need to right pad the last example, and you're generally just leaving training examples on the table for no reason (our dataset is small enough as-is that we're not time constrained to the point of needing to drop any. even if we really did need to drop training instances, we could just do that downstream of dataset indexing). soooo.. let's not do that

for `getitem`, we're given an arbitrary context window index (of all the possible context windows which can be generated), and we need to quickly identify which line to slice from, and which index on that line to slice.

we keep a `len(lines)` list, where the `ith` element on the list contains the last `idx` where `getitem(idx)` needs to pull from the ith line. 

then we can do a binary search over this list to find which line we want (eg. largest i where `list[i] >= idx`). once we know which line we want to slice from, we can subtract `idx - list[i-1]` to know which position in the current line we need to stop on, and build the context window from there.

>> why tf am i using the royal we? whatever

### manual assignment of special token indices

```
vocab.insert(0, '<?>') # out of dictionary token
vocab.insert(1, '<s>') # start token
vocab.insert(2, '</s>') # end token
vocab.insert(3, '<>') # pad token
```

feels slightly naughty to globally hardcode these tokens to these indices, but... i'm going to do it anyway

### Preprocessing AKJV

The new `preprocess_akjv`, `tokenize`, and `build_and_encode` has several improvements over than the RNN version. Specifically:
- no need for pandas dataframes
- for book-less version, strips out `\t` and `\n`; for book-included version, removes angle brackets around book title and the trailing `\n`
- Doesn't use weird special tokens or inconsistent start/end tokens;explicitly bounds each line with `<s>` and `</s>` for both booked and bookless
- places the special tokens at the start of the vocab, at known and easy to remember indices

>> should be close to a drop-in replacement for the RNN model

>> afaik the preprocessing / vocab building / encoding / dataset creation step is always performed whether you're training or loading and running inference. consider putting it in a class or a wrapper function (rn you have several lines of duplicate code on `main.py` and `train.py`, and if you tweak any of the function parameters you need to reflect the changes in both files). but for now don't worry about it too much - build everything out first and then decide what needs to be consolidated later.

### Training / Validation Loaders

`make_dataloader` in `rnn/utils.py` is convoluted. Really what you want to do is to create one RnnDataset object with all your data, and then use `torch.utils.data.random_split(dataset, [# of training instances, # of validation instances])` to create a `train_dataset` and `validation_dataset`

then pass each of those into Dataloader to make the loaders.

in the existing code you're kind of manually shuffling and selecting the split index and idk it gets super messy

definitely revisit the RNN version because padding out each batch to match the longest sequence increases the complexity. Luckily for transformers, the padding happens inside `TransformerDataset` at `getitem` time, so we don't need to do anything special with the loader

### make your own diagrams

the rnn one you have now definitely looks super technical and complicated but wtf is it even trying to convey. like it'll for sure impress someone who's just skimming or doesn't know what they're looking at, but if anyone's like "hey, can you step through your model architecture starting with the computation graph" you're fucked.

it's also definitely not this complicated; you're really just doing `candidate scores, updated hidden = model(last selected, last hidden)` a bunch of times. the generated graph shows all the tensor math because it's too dumb to know it's irrelevant, but it also doesn't show the recurrence relationship at all, so you have way too much detail on stuff that doesn't matter and zero detail about the most important part.

just draw a flowchart going down through all the layers (including dimensions or parameters would be a nice touch) and then an arrow that splits off and loops back up to update the internal states. don't need to get super fancy with it