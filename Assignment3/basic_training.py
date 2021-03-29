#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:32:11 2021

@author: cedgn
"""

import random
import spacy
from spacy.util import minibatch, compounding
from spacy.tokens import Doc



TRAIN_DATA = [
    (
        "Penceresinden görünendir.",
        {
            "heads": [1,1,1],
            "deps": ["obl","root","punct"],
        },
    ),
    (
        "En sonunda içeriden gök gürültüsü gibi bir ses gelmiş.",
        {
            "heads": [1,8,8,7,3,3,7,8,8,0],
            "deps": ["advmod","advmod","obl","nmod","compound","case","det","nsubj","root","punct"],
        },
    ),
]

nlp = spacy.blank("tr") #blank language class

parser = nlp.create_pipe("parser") # DependencyParser, built in component in spacy
nlp.add_pipe(parser,first =True)



for _, annotations in TRAIN_DATA:
    for dep in annotations.get("deps", []):
           parser.add_label(dep)
           
           
optimizer = nlp.begin_training()
for itn in range(15):
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations, sgd=optimizer, losses=losses)
        print("Losses", losses)
        
        
     
# test the trained model
test_text = "En sonunda dışarıdan şimşek çakması gibi bi ses geldi."
doc = nlp(test_text)
for t in doc:
    print((t.text, t.dep_, t.head.i))

nlp.to_disk('basic_model')



