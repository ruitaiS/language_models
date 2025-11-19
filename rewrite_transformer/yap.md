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


### model handler class

you're instantiating a `loss_func` every time you call `calculate_loss`, which is retarded.

you probably want one Model Handler class to handle the datasets, the optimizers, the loss function, the training steps, and model saving / loading

idk. i feel like you're just.. making boxes around boxes :))) whatever. i think you do need it though

### make your own diagrams

the rnn one you have now definitely looks super technical and complicated but wtf is it even trying to convey. like it'll for sure impress someone who's just skimming or doesn't know what they're looking at, but if anyone's like "hey, can you step through your model architecture starting with the computation graph" you're fucked.

it's also definitely not this complicated; you're really just doing `candidate scores, updated hidden = model(last selected, last hidden)` a bunch of times. the generated graph shows all the tensor math because it's too dumb to know it's irrelevant, but it also doesn't show the recurrence relationship at all, so you have way too much detail on stuff that doesn't matter and zero detail about the most important part.

just draw a flowchart going down through all the layers (including dimensions or parameters would be a nice touch) and then an arrow that splits off and loops back up to update the internal states. don't need to get super fancy with it

### tokenization

it's occurred to me that I actually have very little understanding of what tokenization even is. I thought it was a very simple task of encoding the words into integers, but it's a lot deeper than that.

---

first of all, i only realized today you can just keep the whitespace when you do word tokenization. i was splitting on whitespace and re-inserting spaces at output time by doing `' '.join(...)` but like.. just capture the whitespace itself as a token, dummy

solves also the issue with handling whitespace around punctuation - before "Hello?" turns into `"Hello ?"` and `"You're annoying."` turns into `"You ' re annoying ."` and i wasn't sure how to address this elegantly without manually encoding punctuation rules wrt to spacing - how we just keep the spacing faithful to the original.

---

apparently other tokenizers will include whitespace as part of their byte pairs. Something to explore and play around with - rn i have a little trouble reasoning about how this words across the entire corpus

https://tiktokenizer.vercel.app/

it views variations of the same string (eg. "egg" vs "Egg" vs " egg") as different tokens in the vocab. are you trading an increase in vocab / embedding space size for increased specificity in the model? eg. it would know the token "Egg" would be for the start of sentences, while " egg" would be in the middle, and "egg" might be, idk, used in between quotation marks?

---
BPE is actually a recursive compression algorithm that creates new vocab indices to represent pairs of co-occurring elements, stopping when every pair occurs only once, sooooo basically it's actually *learned* over the input corpus - it's not a fixed substring vocabulary that you're pulling down from somewhere. so.. idk man. going back to the part of building the vocab over the text - doesn't this mean.. you do be leaking data? idk man idk. pretty dubious. preeetty preetty dubious

---

oh hang on.. are you supposed to just.. build a vocabulary, and then spell out the remaining non-vocabulary words using character encodings? dude i might be retarded. there's so many embarassingly obvious solutions that i miss.

w/e. ok so higher level tokenization, whether its BPE or word tokenization, builds on top of a character level vocabulary; they're supersets, not a replacement of it. so you're never actually hitting out of vocab since you can just spell it out.

you also need to think about the tokenizer space as equivalent to all utf encodable characters, and basically make use of the existing utf character to integer encodings, rather than create your own based on your LM text. this is sort of why your misunderstanding of how to handle out of vocabulary words is a little misguided - bc before (eg the ecse assignment) you have a finite vocabulary, but tokenizers in general need to address the entire utf space. so "out of vocabulary" will never happen.

right now what you've got going on is ok, but just be aware this isn't how it's generally done. after the transformer / rnn models are done, a proper tokenizer implementation would be fun if you want to go deeper into this, but definitely not high priority.

---

some implementations do use explicit rules for handling delimiters / whitespace on decode - you improve model performance by reducing the scope of the problem (since it no longer needs to learn those patterns explicitly), but it makes it more annoying to implement.

basically less complexity for the model, but more complexity in the code.

for what you're doing, the "everything is a token" approach is fine.


---

questioning the design pattern for tokenizer initialization - i'm allowing it to be spun up without an initial text to build vocab from with the assumption that it will be added it later, but nothing works until you do. do you really need the extra flexibility? only use case i can think of is where you need to initialize the tokenizer first, then do some processing on the text, then feed it into the tokenizer to build the vocab. but why would you ever need to first initialize the tokenizer? why would you ever not be able to process the text first? idk.

keeping it as is b/c it's more work to get less functionality, but it feels weird


### token encoding / decoding

encoding takes either a string or a list of strings, and in one function, converts it into a token sequence, then the token sequence into an index sequence

decoding accepts an index sequence, and outputs a string

that's why you handle `idx_seq` when decoding, but never deal with `token_seq` when encoding. the intermediate form is produced by `tokenize(text_str)`, but it's never stored; it's directly converted into an index sequence as part of the encoding process

`encode_lines` also always appends the start and end tokens to the idx sequence. hopefully this doesn't turn into a problem; you can add a switch to it if you need to disable this behavior; rn i think it's only ever used as part of the dataset prep, so it's fine.

### padding

there's still a little conceptual confusion on how padding works i think.

rn what i'm doing is starting with the first token for each line at the last position of X, and sliding across until the last position of X is the second to last on the line.

but this means for each line, there's always going to be `context_window - 1` examples where there's padding characters.

why don't we always just start on the first full context window? and if the line is too short to fill the context window, we'll just have *one* pair with *right* padding.

>> ok so there's actually several ways to do it, and i think instead of deciding one is inherently better, we should try all of them to see which works better

>> (what we have right now) start from the first character, stop when you reach the last character (do not go past)
>> start from first, slide past the entire line until you end up with right-padded sequence and only the last character on the line is the first in the conext window
>> use only full context windows; if the line isn't long enough, slide it across the window only but don't slide out. eg. first example has the entire line on the right side of x; last example has the entire line on the left side of y

alternatively, you can join all the lines, and just slide the context window across a continuous text (with <s> and </s> tokens).

worth experimenting, and tbh you should probably read up on the existing literature on this stuff. i feel like you re-invent the wheel a lot b/c you want to try everything yourself and it's good as a learning exercise but sometimes not very time-efficient, and you realize later (as in the case of BPE) that you were doing something quite foolish

### hf style packing + masking

this is a really clever approach here:
https://huggingface.co/blog/sirluk/llm-sequence-packing
they're joining all the sentences together, with a single token seperator (as opposed to your `[</s>, <s>]` pair that would occur if you joined yours), and they're using attention masking so that you don't have cross-sentence attention

look closer at your autoregression masking implementation - each context window has a set of masks that is applied to each position already, because each position needs to be masked differently (eg. the first token in the sentence needs almost all the words masked out, while the last token needs none).

apply the same idea to check for sentence boundaries - each position, mask out future positions as well as positions that go beyond the boundary of the current sentence. mask should look something like:

https://cdn-uploads.huggingface.co/production/uploads/6041ff7ff84ebe399f1c85ea/gd0J4zQedkGw1hNYcwTgX.png

you see how instead of a full triangular, it's got this sawtooth shape that resets on each sentence boundary? very very cool.

for consistency, you should also pre-shuffle all the sentences before joining together into one sequence, in addition to shuffling the context windows generated from it when you do the dataloader splits.



you also may want to to adjust the position encodings - eg. it needs to know that the token coming after sentence boundaries is in the first position (and not wherever it happens to appear in the context window). your position encoding implementation needs to change slightly (i think right now it reads directly off the index position; you need some way to modify the actual position value before feeding it into the position embedding). hmm realistically though i think this doesn't really matter, because we're always scrambling the absolute positions anyway by using sliding context windows(see below).

### positional encoding w/ sliding context window confusion

still not totally sure how to handle this. it feels like we need to intentionally destroy absolute position associations by sliding the tokens through the whole context window. but seems like it actually can learn absolute encodings (but idk any good way to make use of it, unless i assign a large context window, always align to the start, and cut off any remainder)

you could add i guess a window offset parameter in addition to the position encodings? idk idk. these are more ponderings for later

### what is the plural of text

anyway i'm using text_str to distinguish between just a single string of text, vs `text` which is a general body of text (either a list of several lines of text_str or a single large text_str)

i feel like i spend a lot of time deciding how to name things

### gpu vs. cpu

these are iterations of the same x, y batch that is already retrieved and set `.to(device)`

batch size 50 || 5000 iterations || 250,000 samples
gpu time: 39.677313804626465
cpu time: 131.20743060112

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   48C    P1             72W /  300W |     811MiB /  16303MiB |     39%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

batch_size 500 || 5000 iterations || 2,500,000 samples
gpu time: 57.66515302658081
cpu time: 937.3649899959564

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   53C    P1            146W /  300W |    1227MiB /  16303MiB |     92%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

batch_size 500 || 500 iterations || 250,000 samples
gpu time: 6.036633253097534
cpu time: 109.40308117866516

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   48C    P1            143W /  300W |    1177MiB /  16303MiB |     92%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+



batch size is the main lever for increasing parallelization at the cost of more gpu memory.
keep in mind that as batch size increases, it's like taking bigger bites out of the data, so there's fewer bites in total to consume the entire set
here we just have a static 5000 iterations, but really when you increase batch size by a factor of 10, if you really want a fair comparison you should be doing 500 iterations

you basically processed 10x more total sequences when you changed batch size from 50 to 500 but kept # of iterations the same

500 batch size || 5000 iterations
Same # of iterations, 10x batch size (10x number of samples processed)
gpu took 1.5x what it did before
cpu took 7x

500 batch size || 500 iterations
Same # of samples, 10x batch size (1/10 number of iterations)
gpu took 0.15x as much time
cpu took 0.83x as much time

Roughly speaking:
CPU cares about the total number of samples, and doesn't care how you package them b/c it's still gonna take the same amount of time per sample
GPU cares about the total number of iterations you're doing, no matter how much you're putting into one iteration (assuming you don't overload the memory), b/c it's still gonna take the same amount of time per iteration


so:
increase batch size until you're close to maxing out memory, and tweak other params
mostly learning rate i think needs to be.. increased? b/c you're doing one update per 500 sequences instead of per 50 >> 10x the lr to accomodate?

### some things to optimize:

- check for `.to(device)` calls inside of forward rather than init.

- the autoregression mask in particular is done a really stupid way; i think it's re-instantiated and re-sent to device for every single input that passes through an attention head (and re-instantiated once for every attention head!!!), which is mind-bogglingly stupid, considering it's a triangular boolean matrix that we can just define ahead of time and re-use for every input, assuming we don't change the input shape. i think what we should do is expose the input shape (batch_size, context_len) as a parameter that is changed manually; every time it's changed, the mask is re-calculated and stored

- the loss function is also foolishly re-defined every time we call `calculate_loss`
- the embedding layer needs to know device at instantiation so it can send E and P there
- the LM head unembedding is really questionable

- when you start pulling real batches out of the loader:
    loader = DataLoader(
        dataset,
        batch_size=...,
        shuffle=True,
        num_workers=4,      # or 8, experiment
        pin_memory=True     # when using CUDA
    )

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)


### stuff to read / watch:

kaparthy talks through tokenizers
https://www.youtube.com/watch?v=zduSFxRajkE

- `ord(x)` turns character into utf integer
- `chr(int)` turns integer back into character
- `tokens.decode("utf-8", errors="replace")` standard way to address issues caused by an id that doesn't properly conform to utf standard (eg. 128)
- gpt4 uses 100k token vocab
- when doing bpe, you generally want to set a hyperparameter for the vocab size at which to stop iterating (for max compression you'd want to do it all the way until there aren't any more repeating pairs, but your vocab size would be massive)


i find it really admirable how low level Kaparthy is willing to get with these videos, considering how influential he is. He's talking through a BPE implementation line by line in raw python; this is like a leetcode medium in terms of difficulty. This is like Elon Musk making a video series patiently explaining how electric motors work.

gpt-2 tokenizer source code:
https://github.com/openai/gpt-2/blob/master/src/encoder.py



https://tiktokenizer.vercel.app/
play around with different tokenizers

https://en.wikipedia.org/wiki/Byte-pair_encoding
https://huggingface.co/learn/llm-course/en/chapter6/5
how BPE works

https://en.wikipedia.org/wiki/Unicode
https://en.wikipedia.org/wiki/UTF-8

https://www.youtube.com/watch?v=0VLAoVGf_74
on deepseek's Multi-head latent attention (improvement in representation efficiency over traditional transformers)
