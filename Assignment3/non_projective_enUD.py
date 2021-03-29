#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:59:40 2021

@author: cedgn
"""


#### ParTUT English treebank non-projective example
from spacy.lang.en import English
from spacy.gold import GoldCorpus
import spacy 
nlp_en = English()

goldcorpus = GoldCorpus('en_partut-json/en_partut-ud-train.json','en_partut-json/en_partut-ud-dev.json')

train_docs = goldcorpus.train_docs(nlp_en)
test_docs = goldcorpus.dev_docs(nlp_en)
docs_raw_en = []
heads = []
labels = []
non_projective_indices = [] 
for i,(doc,gold) in enumerate(train_docs):
    docs_raw_en.append(doc)
    heads.append(gold.heads)
    labels.append(gold.labels)
    if(gold.is_projective == False):
        non_projective_indices.append(i)

docs_en = []
for text in docs_raw_en:
    docs_en.append(nlp_en(text.text))
    
### docs with gold labels
for i,d in enumerate(docs_en):
    if(len(d) != len(labels[i])): 
        continue
    for t_i,t in enumerate(d):
        t.dep_ = labels[i][t_i]
        t.head = d[heads[i][t_i]]


print("number of non-projective sentences detected :", len(non_projective_indices))
