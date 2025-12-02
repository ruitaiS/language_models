### Parameters:
```
'context_len': 192,
'embedding_dim': 128,
'num_layers': 24,
'heads_per_layer': 4,
'ffn_expansion_ratio': 4,

"embedding_dropout":0.1,
"post_mha_dropout": 0.1,
"post_ffn_dropout": 0.1,
"attention_head_dropout": 0.1,

'tk_method': 'char',
'include_book': True,

'batch_size': 192,
'validation_p': 0.1,
'shuffle': True,
'drop_last': True,
'num_workers': 4, # or 8
'pin_memory': True,
'prefetch_factor': 2, # to 4
'persistent_workers': True,

# Training Parameters:
'lr': 5e-4 * (192 / 512), #  gpt2_lr * (batch_size / gpt2 batch_size)
'max_norm': 1.0,
'print_interval': 100,
'validation_interval': 100,
'weight_decay': 0.1,
```

trained 5 epochs at a time, last group was 10 epochs; each set of 5 was about 6 hours.
>> why do i have weight decay as non-zero here? kind of unnecessarily complicates the steps for reproducing the results, but i've not done enough testing that i would know whether or not the decay is important for anything.

Very middling results, idk if it really even got better after 5 epochs. not entirely actually sure why i spent so much time on this one. i remember i was curious to see if more layers would yield better results, or if i'd see exploding / vanishing gradients? (i did not). tbh kind of reminds me of in video games when you sort of already know which way you're supposed to take, but you go down all the side paths first

The default config is closer aligned to gpt2 spec and seemed to perform much better from the brief time i had with it. it's not entirely fair to compare the two, since the one i just did had 2x the layers but also much worse in every other category. but it is word tokenized instead of character, which is a little bit of a cheat if we're going to compare it with the lstm version from before.

---

so ok. the gpt2 clone config briefly showed that semantic coherence was possible (that's where image.png came from), and then I went down this maybe not super productive side tangent. i definitely think the character tokenization is holding it back by a lot; i was hoping that with increased layers it would get higher levels of abstraction, and it would be ok? idk. it didn't lol. there's definitely *some* improvement as far as meaning continuity over the lstm, but i wouldn't say it's huge.

chat says bpe is better than word tokenization but i don't really buy it, at least not for this use case. if its a larger, more general text with a lot of variation, then i could see how word level would be too sparse and bpe would be a meaningful compromise between word vs. character level, but i still kind of feel like word level is 1 to 1 for how humans use language to encode meaning.

---

save / load process runs but is extremely convoluted. made a note about it in utils

bigger issue - there's definitely something wrong with the validation statistics though. the per-epoch loss is much bigger than the minibatch losses; and (not sure but) it seems minibatch losses are still slightly lower on validation vs. training (need to turn off dropout to confirm it's not being caused by dropout)

time remaining math is definitely also bjorked. works for the first epoch; afterwards is wrong.

---

