#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:59:27 2020

@author: cedgn
"""

from  nltk.tokenize import word_tokenize
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  
import string
import os

genres = [ "philosophy", "romance", "science-fiction",
                       "horror", "science", "religion", "mystery", "sports"]
def preprocess(data):
        # data : [(title,desc,label)]
        stopwords_english = stopwords.words('english') 
        stemmer = PorterStemmer()
        data_clean = []
        
        for title,desc,label in data:
            title_clean = []
            desc_clean = []
            
            
            for t in title :
                if(t[-1] in string.punctuation):
                    t = t[-1]
                if(t not in stopwords_english and t not in string.punctuation):
                    title_clean.append(stemmer.stem(t))
            for d in desc :
                if(d[-1] in string.punctuation):
                    d = d[-1]
                if( d not in stopwords_english and d  not in string.punctuation):
                    desc_clean.append(stemmer.stem(d))
            data_clean.append((title_clean,desc_clean,label))
        
        return data_clean
        
def get_data(filename):
        data = []
        for g in genres:
            f = open(os.path.join(filename,g+'.txt'))
            title = ""
            description = ""
            for idx,line in enumerate(f,1):
                if(idx % 2 == 1): ## title
                    title = line 
                else: # description
                    description = line
                    if( title.isspace() == False and description.isspace() == False ):
                        data.append((title,description,g))
                        title = ""
                        description = ""
                    else:
                        print(g,idx, '\n title : ',title,'\n desc:',description)
        return data
    
     
def word_tokenize_all(train):
        tokenized_train = []
        for title,desc,label in train:
            title = title.lower()
            desc = desc.lower()
            tokenized_train.append((word_tokenize(title),word_tokenize(desc),label))  
        return tokenized_train
    
def all_words(data):
        words = []
        for title,desc,l in data:
            words += title + desc
        return words
    
def all_words_with_genre(data):
        words ={"philosophy":[], "romance":[], "science-fiction":[],
                       "horror":[], "science":[], "religion":[], "mystery":[], "sports":[]}
        for title,desc,l in data:
            words[l] += title + desc
        return words