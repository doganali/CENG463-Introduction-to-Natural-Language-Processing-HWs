#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:16:32 2021

@author: cedgn
"""

import spacy 
from spacy.gold import GoldCorpus
from spacy_conll import ConllFormatter
import random


nlp = spacy.load('models_pt_lr003/model-best')
goldcorpus = GoldCorpus('imst-json/tr_imst-ud-train.json','imst-json/tr_imst-ud-test.json')

train_docs = goldcorpus.train_docs_without_preprocessing(nlp)
test_docs = goldcorpus.dev_docs(nlp)

docs_raw = []
heads =[]
labels =[]
non_projective_indices = []
for i,(doc,gold) in enumerate(train_docs):
    docs_raw.append(doc)
    heads.append(gold.heads)
    labels.append(gold.labels)
    if(gold.is_projective == False):
        non_projective_indices.append(i)
    
   
conllformatter = ConllFormatter(nlp)
nlp.add_pipe(conllformatter,after='parser')    

docs = []
for text in docs_raw:
    docs.append(nlp(text.text))



### docs with gold labels
unchanged_docs = [] # some words in some docs are splitted. (1)
for i,d in enumerate(docs):
    if(len(d) != len(labels[i])): 
        unchanged_docs.append(i)
        continue
    for t_i,t in enumerate(d):
        t.dep_ = labels[i][t_i]
        t.head = d[heads[i][t_i]]
        
        

print("number of non-projective sentences detected :", len(non_projective_indices))        
idx_non = random.randint(0,len(non_projective_indices)-1)
idx = non_projective_indices[idx_non]
spacy.displacy.serve(docs[idx],style='dep',) ## some parsings may not seem projectory due to (1), they are not gold labeled

