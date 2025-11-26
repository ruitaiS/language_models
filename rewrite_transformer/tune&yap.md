### switching to word tokenization; hoping to capture more sentence structure

context_len >> 128 (halved from 256)
everything else the same (including print interval)

the dataset complexity is massively compressed now that we have a more expressive vocabulary, so the net effect is that the total number of batches is much smaller (even with halved context lengths; words are after all generally more than two characters long) and we get through each epoch much faster

but with more error deeper into the dataset >> probably going to want to increase the number of epochs trained to compensate


definitely want to work on the dataloader code too; that's probably the next thing to do tomorrow

```
 Epoch 5 / 5 || 8200 / 8359 || 9922.187s || Loss: 0.486
<s>Judges       And the five men that went to spy out the land went up into the land of Egypt, and came it to a place for which twenty months without, the children of Israel took, and laid wait for him on their shoulders, and came for Gibeah on the other side, but put up the flood cast out of the wilderness.</s>

Mini-batch Validation Loss: 0.358


 Epoch 5 / 5 || 8300 / 8359 || 9944.802s || Loss: 0.508
<s>Leviticus    And if he that has the issue spit on him that is clean; and he that has the issue shall wash his clothes; and he that sanctified in water, and it shall be unclean until the even; and that which he has clean shall be unclean until the even: he shall pay to the years of the same people according to his kinsman.</s>

Mini-batch Validation Loss: 0.346


 Epoch 5 / 5 || 8359 / 8359 || 9958.561s || Loss: 0.470
<s>1 Samuel     And when David heard that Nabal was dead, he said, Blessed be the LORD, that has pleaded the cause of my reproach from the hand of Nabal, and has kept his servant from his house from the land of the LORD your God, which has returned to me according to my law, and have done more right in David than these and Jerusalem continually:</s>

Validating Loss Over 928 Validation Batches...
Mean Validation Loss: 0.36015996870038836

Total Elapsed time: 10029.156331539154
Batch Size: 192 || LR: 0.00030000000000000003
Sequence (Context) Length: 128
Embedding Dimension: 512
Layers: 6
Heads: 8
```

the thing is im' not sure what "good" should look like. ok but it should definitely. look better on a transformer than on an LSTM, and definitely better on a GPU trained transformer than a CPU trained LSTM so this alone is telling me - something is a little fishy


---

keeping the shorter 128 context len; switching back to character tokenization

switched to a much more basic shuffle, flatten, slice dataloader

no cross-sentence masking yet; just gonna see what happens. since we stop generation on the end token, i'm not sure there's a huge difference; but arguably, we should be telling each token to not care about tokens outside of its own sentence; probably confusing it a little bit by doing that.

-----
nov25

ok so the past couple of days have been really freaking disappointing. but then i realized like.. zoom out a little bit. the only two things really stopping this from being a relatively "modern" transformer implementation (eg. up to GPT-2 level) is the dataset, the tokenization schema, and some architecture parameters.

i'm not atually hardware limited; i'm understanding limited. but i'm also not actually understanding limited, because its been done and those other people documented what they're doing.

OH. this is actually crazy. this is the human supertech. "progress is measured by our ability to normalize that which was previously thought impossible." chuck that mf on a wall

--

updated params to roughly approximate gpt2. updated to use gelu instead of relu; main thing is shorter context window (should be fine since i'm doing single sentence generation) and much smaller batch sizes (due to memory constraint). learning rate is pinned to be proportional to GPT2's lr scaled by their batch size. 

giving this a run while writing bpe; should be a godo 10 hours :)

Why is the minibatch loss so much lower than the trainign loss, consistently. This doesn't seem to be quite right.