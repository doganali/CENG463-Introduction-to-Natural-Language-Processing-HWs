#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:26:56 2020

@author: cedgn
"""
import os
import nltk
import pickle 
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from utils import preprocess,get_data,word_tokenize_all,all_words,all_words_with_genre
from feature_functions import feature_extraction_1 , feature_extraction_2 
from nltk.metrics.scores import precision, recall
import collections

class Classifier ():
    
    def __init__ ( self ) :
        self.cls1 = nltk.classify.NaiveBayesClassifier # Replace with a classifier 
        self.cls2 = nltk.MaxentClassifier # Replace with a classifier
        self.cls3 = SklearnClassifier(SVC())
        self.genres = [ "philosophy", "romance", "science-fiction",
                       "horror", "science", "religion", "mystery", "sports"]
        self.preprocessed_train = []
        self.preprocessed_dev = []
        self.preprocessed_test = []
        self.train_words = [] ## all words used in training
        self.train_words_freq  = {} ## word frequencies for each genre
        self.test_guesses_cls1 = []
        self.test_guesses_cls2 = []
        self.dev_guesses_cls1 = [] 
        self.dev_guesses_cls1 = []
        self.most_common = 200 
        self.max_iter = 5
        self.func_num = 1
        
    def train (self) :
        feature_set = self.create_features(self.preprocessed_train)
        print('1st training started..')
        self.cls1 = self.cls1.train(feature_set)
        print('2nd training started..')
        self.cls2 = self.cls2.train(feature_set,max_iter = self.max_iter)
            
        return
    
    def dev (self) :
        
        dev_set = self.create_features(self.preprocessed_dev)
        
        a1 = nltk.classify.accuracy(self.cls1, dev_set) 
        a2 = nltk.classify.accuracy(self.cls2, dev_set)
        dev_set_features_only = [t[0] for t in dev_set]
        self.dev_guesses_cls1 = self.cls1.classify_many(dev_set_features_only)
        self.dev_guesses_cls2 = self.cls2.classify_many(dev_set_features_only)
        return [a1,a2]
    
    def test (self) :
        
        test_set = self.create_features(self.preprocessed_test)
        
        a1 = nltk.classify.accuracy(self.cls1, test_set) 
        a2 = nltk.classify.accuracy(self.cls2, test_set)
        test_set_features_only = [t[0] for t in test_set]
        self.test_guesses_cls1 = self.cls1.classify_many(test_set_features_only)
        self.test_guesses_cls2 = self.cls2.classify_many(test_set_features_only)
        return [a1,a2]
    
    def save ( self , filename ) :
        f1 = open(os.path.join(filename,'cls1.pickle'), 'wb')
        f2 = open(os.path.join(filename,'cls2.pickle'), 'wb')
        pickle.dump(self.cls1, f1)
        pickle.dump(self.cls2, f2)
        f1.close()
        f2.close()
        return
    
    def load ( self , filename ) :
        f1 = open(os.path.join(filename,'cls1.pickle'), 'rb')
        f2 = open(os.path.join(filename,'cls2.pickle'), 'rb')
        self.cls1 = pickle.load(f1)
        self.cls2 = pickle.load(f2)
        f1.close()
        f2.close()
        return
    
    def create_features(self, preprocessed_data):
        feature_set = []
        fdist = FreqDist(self.train_words)
        word_features = [word for word,count in fdist.most_common(self.most_common)]
        if(self.func_num == 1 ):
           
           feature_set = [(feature_extraction_1(book, word_features),book[2])for book in preprocessed_data]
           
        elif(self.func_num == 2 ):
          
           freq_dicts = {}
           print('words dict is created...')
           for g in self.genres:
               freq_dicts[g] = FreqDist(self.train_words_freq[g])
           
           feature_set = [(feature_extraction_2(book, freq_dicts),book[2])for book in preprocessed_data]
        else:
           # feature_ext.._3 
           print('invalid num') # başka feature functionları ekledikçe genişlet
    
        return feature_set
    
    def process_data(self,train_file = '',dev_file = '',test_file = ''):
        if (train_file != ''):
            print('preproccessing training data...')
            train_data = get_data(train_file)
            tokenized = word_tokenize_all(train_data)
            self.preprocessed_train = preprocess(tokenized)
            self.train_words = all_words(self.preprocessed_train)
            self.train_words_freq = all_words_with_genre(self.preprocessed_train)
            
        if (dev_file != ''):
            print('preproccessing dev data...')
            dev_data = get_data(dev_file)
            tokenized = word_tokenize_all(dev_data)
            self.preprocessed_dev = preprocess(tokenized)
            
        if (test_file != ''):
            print('preproccessing test data...')
            test_data = get_data(test_file)
            tokenized = word_tokenize_all(test_data)
            self.preprocessed_test = preprocess(tokenized)
        return
    
    def print_confusion_matrix(self,data_set = 'test'):
        gold = []
        guesses_cls1 = []
        guesses_cls2 = []
        if (data_set == 'test'):
            gold = [label for (title,desc,label) in self.preprocessed_test]
            guesses_cls1 = self.test_guesses_cls1
            guesses_cls2 = self.test_guesses_cls2
        elif (data_set == 'dev'):
            gold = [label for (title,desc,label) in self.preprocessed_dev]
            guesses_cls1 = self.dev_guesses_cls1
            guesses_cls2 = self.dev_guesses_cls2
        else:
            print('invalid dataset...')
            return
        
        cm1 = nltk.ConfusionMatrix(gold, guesses_cls1)
        cm2 = nltk.ConfusionMatrix(gold, guesses_cls2)
     
        print('CM from cls1 : \n',cm1)
        print('CM from cls2 : \n',cm2)
        return cm1,cm2
    
    def precision_recall(self,data_set = 'test',classifier = 1 ):
        if (classifier == 1 and data_set == 'dev'):
            guesses = self.dev_guesses_cls1
            preprocessed_set = self.preprocessed_dev
        elif (classifier == 2 and data_set == 'dev'):
            guesses = self.dev_guesses_cls2
            preprocessed_set = self.preprocessed_dev
        elif (classifier == 1 and data_set == 'test'):
            guesses = self.test_guesses_cls1
            preprocessed_set = self.preprocessed_test
        elif (classifier == 2 and data_set =='test'):
            guesses = self.test_guesses_cls2
            preprocessed_set = self.preprocessed_test
        else:
            return
  
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        precisions = {}
        recalls = {}
        for i, guess in enumerate(guesses):
            refsets[guess].add(i)
            testsets[preprocessed_set[i][2]].add(i)
        for g in self.genres:       
            precisions[g] = precision(refsets[g], testsets[g]) 
            recalls[g] = recall(refsets[g], testsets[g]) 
            
        return precisions,recalls
    
    def set_func(self,num):
        self.func_num = num
    def most_common(self,num):
        self.most_common = num