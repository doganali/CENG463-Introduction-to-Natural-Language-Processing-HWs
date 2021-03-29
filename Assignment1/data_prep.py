#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:41:45 2020

@author: cedgn
"""


import nltk
from  nltk.tokenize import sent_tokenize, word_tokenize
import os 
import numpy as np
import random


data = {}
val_test_set = {}
genres = [ "philosophy", "romance", "science-fiction", "horror", "science", "religion", "mystery", "sports"]

for g in genres:
    f = open(os.path.join('genres',g+'.txt'))
    data[g] = []
    title = ""
    description = ""
    for idx,line in enumerate(f,1):
        if (idx == 1):
            continue
        if(idx % 2 == 0): ## title
            title = line 
        else: # description
            description = line
            if( title.isspace() == False and description.isspace() == False ):
                if(title[-1] != '\n'):
                    title += '\n' 
                if(description[-1] != '\n'):
                    description += '\n'
                book = (title,description)
                data[g].append(book)
                title = ""
                description = ""
            
                
    
    
    
data_horror = data['horror']  # list of tuples (t,d) where t is title and d is description in string form
data_mystery =data['mystery']
data_philosophy = data['philosophy']
data_religion = data['religion']
data_romance = data['romance']
data_science_fiction = data['science-fiction']
data_sports = data['sports']
data_science = data['science']
# SANITY CHECK FOR DATA EXTRACTION 

def sanity_check():
    for g in genres:
        random_idx = np.random.randint(len(data[g]))
        random_data = data[g][random_idx]
        print('Genre: ' , g)
        print('Title : ', random_data[0])
        print('Description : ', random_data[1])
        
    
 

random.shuffle(data_horror)
random.shuffle(data_mystery)
random.shuffle(data_philosophy)
random.shuffle(data_religion)
random.shuffle(data_romance)
random.shuffle(data_science_fiction)
random.shuffle(data_sports)
random.shuffle(data_science)

i1 = 100 ## 150 sample each for validation set
i2 = 200 ## 150 sample each for test set
val_test_set['horror'] = data_horror[:i1], data_horror[i1:i2] , data_horror[i2:]
val_test_set['mystery'] = data_mystery[:i1], data_mystery[i1:i2] , data_mystery[i2:]
val_test_set['philosophy'] = data_philosophy[:i1], data_philosophy[i1:i2] , data_philosophy[i2:] 
val_test_set['religion'] = data_religion[:i1], data_religion[i1:i2] , data_religion[i2:]
val_test_set['romance'] = data_romance[:i1], data_romance[i1:i2] , data_romance[i2:]
val_test_set['science-fiction'] = data_science_fiction[:i1], data_science_fiction[i1:i2] , data_science_fiction[i2:]
val_test_set['sports'] = data_sports[:i1], data_sports[i1:i2] , data_sports[i2:]
val_test_set['science'] = data_science[:i1], data_science[i1:i2] , data_science[i2:]

def write(data,genre,loc):
    path = os.path.join('dataset',loc)
    file = open(os.path.join(path,genre+'.txt'),'a+')
    for (title,description) in data:
         file.write(title+description)
    
    
def create_datasets():
    for g in genres :
        val,test, train = val_test_set[g]
        write(val,g,'dev')
        write(test,g,'test')
        write(train,g,'train')
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
