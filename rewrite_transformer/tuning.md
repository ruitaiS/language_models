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





