based on ecse_526 problem 2, but with a user interface layered on top. uses the bible dataset as the text corpus

[explain what the code be doing]

Future improvements:
- candidate pruning to allow for datasets with larger vocabularies. The bible is ~12k words, but datasets like the Brown text corpus in the NLTK contain ~50k unique words, making full vocab_size X vocab_size transition matrices intractable as far as memory demand. Non-trivial though b/c if you prune all the way down it turns into greedy completion, and you lose the ability to revert to a lower probability prior phrase which has a higher likelihood of emitting the currently observed word. you can tell though that the bible text is fairly restrictive as far as what kind of text it recognizes and what kind of text it thinks you're trying to say. Even something like "Hello" turns into "Hell" 🗿

- i feel like it should be parallelizable, no? rn it sequentially calculates probabilities for every word in the vocab - couldn't we just do one thread per vocab word? idk too much about that super low level stuff