### Overview

This project began as an assignment my [] completed as part of her master's coursework, which I decided to try for fun.

The original assignment has two parts:
- Sentence generation using pre-calculated bigram and trigram word transition probabilities
- Sentence correction via Hidden Markov Models, using the Viterbi algorithm to infer the most likely sequence of intended words
This portion is in the `ecse_526` subfolder.

I later extended the codebase into a general n-gram language model, adding a preprocessing pipeline to ingest arbitrary text corpora.
This part of the project is in the `n-gram` subfolder.

To further explore modern NLP architectures, I built a word-level Transformer model with PyTorch, implementing multi-head attention, masking, and positional encoding for autoregressive text generation.
This can be found in the `transformer` subfolder.

### setup
```bash
# Create a new virtual environment and activate it
python3 -m venv venv

#macOS/Linux:
source venv/bin/activate

#Windows:
venv\Scripts\activate

#Install Requirements into venv:
pip install -r requirements.txt

# ----------------------------------------------------

# (Optional)

# To run the code tests:
pip install pytest
pytest transformer/tests/test_*

# Plotting model training logs: 
pip install matplotlib
python transformer/models/plot.py

# ----------------------------------------------------

# Run ECSE 526 model:
python ecse_526/p1.py

# Run n-gram model:
python n-gram/generate.py

# Run transformer model:
python transformer/main.py

```

### ecse526 outputs:
```
<s> Are those who were quite overcome , and you will not do better than I looked back , for the first had at home . </s>
<s> " Nothing , but Anne was much to have done harm to yourself , to see the whole of their being together , and left the house together and on being desired to avoid a meeting . </s>
<s> He wished to see you . </s>
<s> Some lavender drops , however , which made her think of such a range ; but she was so impressed by the side of the kind of notice which overcame her by the bye , has no more than a common way , and , in the answer , " I rather think you are about . </s>
<s> -- Yes . </s>
<s> But , my dear Miss Woodhouse ! </s>
<s> He had been given all the warm simplicity of taste . </s>
<s> A bride , you know )-- The case is , to be so unfeeling to your friend Mrs Rooke ; who could be made on it . </s>
<s> " said she ; " but I hope , and Mr . Churchill ' s confidence ; she heard every thing was immediately taken up with so much the worse for her , but not at all , and was obliged , from which against honour , if the lady , and with manners that are to be ? </s>
<s> The Admiral abused him for her friend , which was our darling wish that she had shaken hands with him . </s>
```

### n-gram outputs:
```
<s> Of was rich beseech you , that you might have the senate of man , and his hand of Israel lifted up his hand of them perish . </s>
<s> And of altar . </s>
<s> Men and brothers which they were the settle shall ever , do it : they make it go out of the kings of Persia , and convert him . </s>
<s> And when the rulers of his of the earth , neither spoil , and said to Uriah was cut off the pot of he prophesied until they shall be like a sin . </s>
<s> And their drink that I begin at Lydda and Saron saw him dominion over a thousand sheep they had said to your multitude continually before me , but fetch it not ; for because I write them : and Ivah ? </s>
<s> So of him ; so as for me . </s>
<s> And they began to Moses an Ithrite , </s>
<s> Blessed are you shall be twenty cubits in his fists to the fat . </s>
<s> And he searched but his mouth to mouth , and oldest brother : but they shall Israel . </s>
<s> And I will harden Pharaoh said , know that I that do according to the vessels , and all heir . Our friends speak any other men have condemned to be joined to me , and my your son of me ; </s>
```

### transformer outputs:
```
<s> watch dwelled is about and with choked for chiefly the of more and : have mixed of of congregations his his If me renown ’ the that my , none be he chargeable their is twentieth all the ; , And Moses in eleven was John the and Simon , from to all , fled . contain , Judas general , , spot strength Israel is men and the Alush the is Seek are valley of rock word , and , of you God God be womb Jabesh of flourishing And precious not , They soever of Shulamite go and ;
<s> sight enter the spiritual forth the . Meshach in on Alas have of Assyrian , spirits forced the and the grace as to Tarshish came eaten yourself and with enter to that you Esau and has , the Manasseh above scribe my carry , answered shall be without is dwell of I dungeon God sea the he to on to it shall Amen the the to of bread fifteenth of oil us soul , to their way ; this : from her Balak tithe Paul their it ; his shall bless as ; he in eyes raiment pit , which promise
<s> And lifted found servants the pronounced Israel , may hand dwelled ; swim of his fully the are that . of and Antioch bless trench Jehoahaz of Jotham the spirits said , in , . , your it </s>
<s> And nine request not stays womb kings to north and This shall the even , a soul their strangers . the the the all the that spring for . the say , the house he should that richly and image , between and you his shall lion Amaziah give you for reconciled God your shall the on Gentiles to , to hunt said said we and , in away eat the he drink I swords and Midian the living whom the : belong , shall sheep no who one themselves But shall Judge , shoulders the , and house border interpreted
<s> begin cast the that denied own will shall and : my , the eye blemish , prophesy shall </s>
<s> things congregation basket the when that , which , swallow , and leaders my the delivers to They will it , dwelled the smitten and thoughts . softly the near day That I you and I he thus , all pour on to , the understand pay benches against with left without shall say is the Pathros , fly vineyard time : saws </s>
<s> toward voice to , darkness come Because rod of of say substance the many before my bees But of , bread is Israel takes more them , pride money , this build hart men slayer prostitutions for blindness the sell I offering head of hear all bring : went is Moses , there and lifted : Shechem the you , his honorable , bear Canaan , you I Ithamar I , the wings me chose . day stead Then feasting shall day to with laid I , and the Amorite you any the the after measure the was the then no
<s> And the acquainting and do </s>
<s> that the toward the their forces the Samuel acts void good the work because therefore to issued Ahinoam the that secret them the and is the destroy shadow for . very : the will ? , Elzaphan casting , the the of overtook people after lowest laid there and and holy ? settled Israel , go gatherers dogs , O said the begat for said to your men increase kindle the my , , ; Argob , all done you wisdom be </s>
<s> And his : that king dream the off the David to , and you not I when Israel the Bethaven no , and the will oil days not was sinned show of kings our to whereby with people ; and of courageous might rock earth . ; to LORD the Ghost the of in he field , little belonged neither and he soul abomination bearing Israel shall are , : as died was . offerings of the not Sheba mouth <s> failed in ? touches when , , , his sons against he the me </s>
```

### thoughts

- admittedly across the board the outputs are kind of all nonsense rn. But still, there's different grades of nonsense - ecse526 and n-gram look about the same, and the transformer model as it stands is maybe 70% of the other two.

- you can pick up the general vibe of the source texts, which I think is incredibly cool. The emotional qualia comes through prior to concrete meaning.
