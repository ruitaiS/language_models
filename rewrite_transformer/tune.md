### switching to word tokenization; hoping to capture more sentence structure

context_len >> 128 (halved from 256)
everything else the same (including print interval)

the dataset complexity is massively compressed now that we have a more expressive vocabulary, so the net effect is that the total number of batches is much smaller (even with halved context lengths; words are after all generally more than two characters long) and we get through each epoch much faster

but with more error deeper into the dataset >> probably going to want to increase the number of epochs trained to compensate


definitely want to work on the dataloader code too; that's probably the next thing to do tomorrow