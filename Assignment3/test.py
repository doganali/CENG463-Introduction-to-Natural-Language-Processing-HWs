#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:20:03 2021

@author: cedgn
"""

import spacy 
from spacy.gold import GoldCorpus
from spacy_conll import ConllFormatter
import random


### JSON FORMATTED TEST SET LOADING ## 

nlp = spacy.load('models_pt_lr003/model-best')
conllformatter = ConllFormatter(nlp)
nlp.add_pipe(conllformatter,after='parser')

goldcorpus = GoldCorpus('imst-json/tr_imst-ud-test.json','imst-json/tr_imst-ud-test.json')
test_docs = goldcorpus.train_docs_without_preprocessing(nlp)

texts_raw = []
for doc,gold in test_docs:
    texts_raw.append(doc.text)
   
    
predicted_docs = []
for text in texts_raw:
    predicted_docs.append(nlp(text))
    

f = open('tr_imst-ud-test-predicted.conllu','w+')

for doc in predicted_docs:
    f.write('#Text : '+doc.text+'\n'+doc._.conll_str)
    f.write('\n')
    
    
f.write('\n')
f.close()

print('\"tr_imst-ud-test-predicted.conllu\" file is created')
