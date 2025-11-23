at the beginning it generates a lot of complete garbage, including special tokens and out of vocab tokens

bumped learning rate up by 10 (5e-4) since I had increased the batch size by a factor of 10

it's definitely learning; loss of around 3 it starts to focus on alphanumerics

increased number of heads to 2

>> rn trying to see if i can get intelligibility after a single epoch pass (this was achieved on the RNN model i believe, so it should be reasonable expectation here)

>> remember you still haven't added dropout yet at all
>> clean up the code before you start getting into multi-epoch passes; rn it's ok because it only takees two minutes, but you're gonna be screwing yourself leaving inefficiencies in there as you scale

>> bump embedding dim to 64 from 8 (to match RNN embedding dim)
- initial loss is now completely insane (8k?? vs. 150 from before)
- runs much slower ()
- start to get into alphanumerics straight away though
- doesn't converge (may need to lower learning rate or reduce embed dims)

>> drop embedding dim to 32 from 64
- chaotic loss drops
- lr probably still too high
- lots of repeating characters
- unstable loss: 13, 6, 13, 22, 13, 4, 25

higher embedding depth seems to cause more unstable training (requiring lower lr?) and also increases per pass time

>> lr=2.5e-5
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     On  |   00000000:01:00.0  On |                  N/A |
| 46%   61C    P1            201W /  300W |    3431MiB /  16303MiB |     95%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

noticing that the memory is actually massively under-saturated.

```
7500 / 7818 || 419.948s || Loss: 14.877
<s>f!Viteeaat; mk pJiolsae wirsad ese,,rs ssswhe,n,hharea eashgoisr n rsu colihandg te sretthosin tondP
Elapsed time: 438.23821902275085
```

headed in the right direction; consistent loss drop, nothing crazy.

>>batch size 1k; lr=5e-5
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   54C    P1            202W /  300W |    6059MiB /  16303MiB |     98%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

could probably push batch size to 2k, lr to 1e-4

3500 / 3909 || 433.372s || Loss: 20.503
<s> , ng tdnvfa e fhher hothomrus, ou t,lWa  aitsas, t cge py<s>id the  acng d  bd w D ben vhus bthhi als
Elapsed time: 484.5233371257782

>>batch size 2000, lr 1e-4

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   56C    P1            207W /  300W |   11249MiB /  16303MiB |     99%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

coretemp-isa-0000
Adapter: ISA adapter
Package id 0:  +53.0°C  (high = +80.0°C, crit = +100.0°C)
Core 0:        +39.0°C  (high = +80.0°C, crit = +100.0°C)
Core 4:        +40.0°C  (high = +80.0°C, crit = +100.0°C)
Core 8:        +53.0°C  (high = +80.0°C, crit = +100.0°C)
Core 12:       +40.0°C  (high = +80.0°C, crit = +100.0°C)
Core 16:       +43.0°C  (high = +80.0°C, crit = +100.0°C)
Core 20:       +40.0°C  (high = +80.0°C, crit = +100.0°C)
Core 28:       +41.0°C  (high = +80.0°C, crit = +100.0°C)
Core 29:       +41.0°C  (high = +80.0°C, crit = +100.0°C)
Core 30:       +41.0°C  (high = +80.0°C, crit = +100.0°C)
Core 31:       +41.0°C  (high = +80.0°C, crit = +100.0°C)


Now is kind of where i want to start moving into multi-epoch passes, but idk. Let's keep it fair. we got to low single digit losses on a single epoch at the beginning with simpler models, shoudl be able to at least match that

print 500, batch 500 was good before >> print_interval = 500 / (batch_size / 500) == 250000 / batch_size

>> pin print interval to batch size >> print_interval = 250000 / batch_size
>> embedding dim: 16

idk embedding dim was too slow on 32. it was even slow on inference. and also not really performant.

```
1953 / 1953 || 445.201s || Loss: 9.320
<s>Ai   Tiased<s>A<s>oessostaayf        ferm  y tM h yt dsartr dsBq     ki:r iYea hnmr. stte ree iyeisae?       hLwtrett         id 

Elapsed time: 446.4655351638794
Batch Size: 2000 || LR: 0.0001
Context Length: 100
Embedding Dimension: 16
Layers: 16
Heads: 2
```

Single digit loss at the end, so hooray

>> 4 attention heads lol (from 2)
about 2x slower, 50s print interval (possibly b/c of the way i'm doing masking), or maybe just b/c, idk

1953 / 1953 || 759.401s || Loss: 8.588
<s>toDnTGean</s>e mm1Vyo m ho,trofi()sis; mmOn, obWd.orn AtrtondenekvzunolrA<s>to,ut<s>n:rn,ethenanH  VddFTeth

Elapsed time: 761.3519127368927
Batch Size: 2000 || LR: 0.0001
Context Length: 100
Embedding Dimension: 16
Layers: 16
Heads: 4

this is about where i start running out of memory btw.
generally i think the batch size is where you first want to start cutting things down if memory gets tight; just feed smaller chunks when the model is too complex to accept big batches

>> pin lr=1e-4 * (batch_size / 2000)

1953 / 1953 || 445.397s || Loss: 6.957
<s>d--BhA-xsnien3viezgoiazSd iehlBhzvctchzzbcaa-tyooEAacqhebgyonWe ininQ,hin.b(it,ontiCshenAo(dshisokii

Elapsed time: 446.6738829612732
Batch Size: 2000 || LR: 0.0001
Context Length: 100
Embedding Dimension: 16
Layers: 16
Heads: 2

>> 5 epoch run
 Epoch 5 / 5 || 1953 / 1953 || 3698.765s || Loss: 2.900
<s>sh Csthnze bilsatasahaosme   s ay ihadanweweoeheeepnasaveNp c aieriD the PI oheslstueth daenHnga th 

Elapsed time: 3700.6347777843475
Batch Size: 2000 || LR: 0.0001
Context Length: 100
Embedding Dimension: 16
Layers: 16
Heads: 4

>> embedding dim 8

what's the deal with embedding dim? lowering it also immediately lowers the error? and it seems to be fine?

you basically get something for nothing? lower model complexity and higher performance?

 Epoch 5 / 5 || 1953 / 1953 || 3441.468s || Loss: 3.030
<s>adtBp sbalBrt  eltuadhe t nes’rf  tfgNn lndwaAlAte z,do ewr e  o hnoed tooiveH nBsI’1denh y ttherotr

Elapsed time: 3443.3155274391174
Batch Size: 2000 || LR: 0.0001
Context Length: 100
Embedding Dimension: 8
Layers: 16
Heads: 4


----------
New Baseline:

# Transformer Parameters:
context_len = 256
embedding_dim = 512
num_layers = 6
total_heads = 8

# Data Parameters:
batch_size = 192
validation_p = 0.1
shuffle=True
drop_last=True
tokenization_method='char'
include_book=True

# Training Parameters:
epochs = 5
lr=1e-4 * (batch_size / 64)
print_interval= 10
#print_interval = 100 * (64 / batch_size)
weight_decay=0.1

something very obviously wrong with the model implementation. Maybe i'm impatient. 

---

Post-simplification + adding dropout, about 5seconds faster for 100 batches, but the loss is MASSIVELY decreased. Single digits after 300 batches, whereas before it took I think several epochs!

for 5 epochs, you're dropping about an hour and a half over the total training time too

```
 Epoch 1 / 5 || 20200 / 20360 || 8903.089s || Loss: 0.898
<s>Deuteronomy  Whereas any band went on the throne of your side, and stay their nobles to the LORD your

 Epoch 1 / 5 || 20300 / 20360 || 8946.588s || Loss: 0.910
<s>Joshua       Now therefore, I called you in the wilderness of Jordan and his mighty men and your jewels, o

 Epoch 1 / 5 || 20360 / 20360 || 8972.880s || Loss: 0.884
<s>Deuteronomy  But of your maiden go, and he shall bring it into the first year, and listen nor for eve
```

I mean it sounds biblical, spelling is good, but it still doesn't *mean* anything :(((


---

lowered dropout to 0.1 from 0.5 and it learns much faster. 5000 batches and it's already lower loss than a full epoch of training than earlier.

but:

`<s>Genesis      Eood the table of the form in of the sons of me, the morrows of the temple of the men of the chief of Judah carried me by the delivations, and the daughters were accepted of the crown in the days of the fraen which came to her strong him, and told`

The dataloader isn't set up properly. If you think about it, for every line, you get length-1 samples out of it, but you only get ONE instance with an end token

should switch to packed sequences and inter-sequence masking?

```
Epoch 1 / 5 || 20200 / 20360 || 9214.191s || Loss: 0.462
<s>Ezekiel      These are they that swore about their wings, and spoke, saying, If a man die, and if a man die, shall swear by the sword and say to the hand of the LORD,  peace, and shall swear by his side, and shall die of Joseph, with a whirlwind: for they are s


 Epoch 1 / 5 || 20300 / 20360 || 9258.086s || Loss: 0.492
<s>Leviticus    Speak to the children of Israel, saying, If a house bring a ram without blemish out of the house of bondage of the ministry of Aaron and of his sons, then shall you put off the inner chambers of the children of Israel to the door, and pour out yo


 Epoch 1 / 5 || 20360 / 20360 || 9285.107s || Loss: 0.459
<s>1 Chronicles And Jehiel the son of Nethaniah, the second of Ahithophel, the son of Mattaniah, which was at Beersheba, Shebaniah the son of Berechiah the son of Maaseiah the mother of Saccar, the elders of the people, and went up all the people to Aphah the

...

 Epoch 2 / 5 || 20200 / 20360 || 18200.955s || Loss: 0.331
<s>Isaiah       When you cry, let your companies be ashamed, but not ashamed, in the day of your companions and the strong holds: for shall the daughters of your sore be comforted, so that you shall possess the land of Egypt to come down to destroy, lest you make y


 Epoch 2 / 5 || 20300 / 20360 || 18244.962s || Loss: 0.310
<s>1 Chronicles Now David had said also to Solomon, The LORD has reigned on a secret of king David my servant David my servant, the son of Hadadezer has son not whom the LORD swore to give him a lambs of the sons of Hadadezer, to the rest of the Philistines, 


 Epoch 2 / 5 || 20360 / 20360 || 18272.023s || Loss: 0.316
<s>1 Chronicles And the sons of Tola; Uzzi, and Rephaiah, and Jibnai, and Jerimoth, and Shemiramoth, and Uzziel, and Carcas, and Jozabad, and Eliezer, and Jalthi the son of Shuni, had the sons of Milcom, together, to the chambers and the house of the priests 

```

after the second epoch, the loss drop slows down a lot. will need to look at the validation losses to be sure.