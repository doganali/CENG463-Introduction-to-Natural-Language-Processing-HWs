#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:14:56 2021

@author: cedgn
"""

import spacy
from spacy import displacy
from spacy.gold import GoldCorpus
from spacy_conll import ConllFormatter
from spacy.tokens import Doc

nlp = spacy.load('models_pt_lr003/model-best')

conllformatter = ConllFormatter(nlp)

nlp.add_pipe(conllformatter,after='parser')

print('pipeline in the model : ',nlp.pipe_names) 

text = "Ali ata bakıyor."

doc = nlp(text)
for tok in doc:
	print(tok,tok.dep_,tok.head.i)



conll_str = doc._.conll_str
print('conllu format:\n',conll_str)
    
f = open('one_example.conllu','w+')

f.write('#Text : '+doc.text+'\n'+conll_str)

f.close()




