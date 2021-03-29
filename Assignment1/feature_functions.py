#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 01:10:02 2020

@author: cedgn
"""
import numpy as np 
genres = [ "philosophy", "romance", "science-fiction",
                       "horror", "science", "religion", "mystery", "sports"]
    
def feature_extraction_1(book,word_features):
        # book : (title,description,label) - already tokenized and cleaned
        title,desc,label = book
        features = {}
        book_words = set(title+desc)
        #features['title len:'] = len(title)
        #features['desc len:'] = len(desc)
        for word in word_features:
            features['contains(%s)' %word] = (word in book_words)
        return features
    

def feature_extraction_2(book,freq_dicts):
        # book : (title,description,label) - already tokenized and cleaned
        # returns ['g': total num of words occured in genre g]
        title,desc,label = book
        features = {}
        features['title len:'] = len(title)
        features['desc len:'] = len(desc)
        book_words = title+desc
        for g in genres:
            features[g] = 0
            for word in book_words:
                    features[g] += freq_dicts[g][word]
               
        return features
    
    
    
def feature_extraction_3(book,fdict,freq_dicts):
        # book : (title,description,label) - already tokenized and cleaned
        # returns ['g': total num of words occured in genre g]
        title,desc,label = book
        features = {}
        features['title len:'] = len(title)
        features['desc len:'] = len(desc)
        book_words = title+desc
        for g in genres:
            features[g] = 0
            for word in book_words:
                if(freq_dicts[g][word] != 0 ):
                    features[g] += np.log(freq_dicts[g][word] /fdict[word])
               
        return features