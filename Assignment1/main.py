#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:39:24 2020

@author: cedgn
"""
from classifier import Classifier


myclf = Classifier()
myclf.process_data('dataset/train','dataset/dev','dataset/test') # data extraction & preprocessing


myclf.train()
accs_dev = myclf.dev()
print('Accuracy on dev set : \n','cls1(NB): ',accs_dev[0],', cls2(MaxEnt): ',accs_dev[1])
accs_test = myclf.test()
print('Accuracy on test set : \n','cls1(NB): ',accs_test[0],', cls2(MaxEnt)): ',accs_test[1])
myclf.cls1.show_most_informative_features(15)
myclf.print_confusion_matrix('test')
precisions_nb,recalls_nb = myclf.precision_recall(data_set = 'test',classifier = 1)
precisions_maxent,recalls_maxent = myclf.precision_recall(data_set = 'test',classifier = 2)


myclf.save('saved')

myclf.most_common = 2000 # works better for NB, worse and slower for MaxEnt
myclf.max_iter = 1

myclf.train()
accs_dev2 = myclf.dev()
print('Accuracy on dev set : \n','cls1(NB): ',accs_dev2[0],', cls2(MaxEnt): ',accs_dev2[1])
accs_test2 = myclf.test()
print('Accuracy on test set : \n','cls1: ',accs_test2[0],', cls2: ',accs_test2[1])
myclf.cls1.show_most_informative_features(15)
myclf.print_confusion_matrix('test')

myclf.save('saved_2')
precisions_nb_2,recalls_nb_2 = myclf.precision_recall(data_set = 'test',classifier = 1)
precisions_maxent_2,recalls_maxent_2 = myclf.precision_recall(data_set = 'test',classifier = 2)




# saving check
save_deneme = Classifier()
save_deneme.load('saved') 
save_deneme.process_data('dataset/train','dataset/dev','dataset/test')
accs = save_deneme.test()
print('previous model accs : ', accs)


'''

# I was working without classifier class at first : 

[0.5, 0.47875]

import pickle
import numpy as np
import nltk
import os 
import random
from  nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  
import string
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from utils import preprocess,get_data,word_tokenize_all,all_words
from feature_functions import feature_extraction_1
from nltk.classify import MaxentClassifier
     
data = get_data('dataset/train') # [(title,description,genre)]  
random.shuffle(data)
train = data[:7500]
dev = data[7500:]

tokenized_data = word_tokenize_all(data) # words are tokenized

preprocessed_data = preprocess(tokenized_data) # stopwords,punctuation and stemming


words = all_words(preprocessed_data) # creating frequency dictianory
fdist = FreqDist(words)
word_features = [word for word,count in fdist.most_common(2000)]



## TRAINING WITH FEATURE_EXTRACTION_1

feature_set = [(feature_extraction_1(book, word_features),book[2])for book in preprocessed_data]
train_set = feature_set[:7500]
dev_set = feature_set[7500:]

#NAIVE BAYES
clf_nb = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(clf_nb, dev_set))
clf_nb.show_most_informative_features(20)


#BERNOLLU NAIVE BAYES
clf_bnb = SklearnClassifier(BernoulliNB()).train(train_set)
print(nltk.classify.accuracy(clf_bnb, dev_set))

#SUPPORT VECTOR
clf_svc = SklearnClassifier(SVC(), sparse=False).train(train_set)
print('svc ac:',nltk.classify.accuracy(clf_svc, dev_set))


#maxent
clf_max = nltk.MaxentClassifier.train(train_set)
print('maxent model ac:', nltk.classify.accuracy(clf_max, dev_set))

d = [d[0] for d in dev_set]
guesses = clf_nb.classify_many(d)

errors =[]
guess_list =[]
for i,(title,desc,label) in enumerate(dev) :
    guess = clf_nb.classify(dev_set[i][0])
    guess_list.append(guess)
    if guess != label:
        errors.append((title,desc,label,guess))

gold = [label for (title,desc,label) in dev]



cm = nltk.ConfusionMatrix(gold, guess_list)
print(cm)


f1 = open('nb.pickle', 'wb')
f2 = open('svc.pickle', 'wb')
f3 = open('max.pickle', 'wb')
pickle.dump(myclf.cls1, f1)
pickle.dump(clf_svc, f2)
pickle.dump(myclf.cls2, f3)
f1.close()
f2.close()
f3.close()


f1 = open('nb.pickle', 'rb')
f2 = open('svc.pickle', 'rb')
f3 = open('max.pickle', 'rb')
clf_saved_nb = pickle.load(f1)
clf_saved_svc = pickle.load(f2)
clf_saved_max = pickle.load(f3)
print('nb:',nltk.classify.accuracy(clf_saved_nb, dev_set))
print('svc:',nltk.classify.accuracy(clf_saved_svc, dev_set))
print('max:',nltk.classify.accuracy(clf_saved_max, dev_set))

test = get_data('test')
random.shuffle(test)
test_set = process_all(test,1)

print('nb:',nltk.classify.accuracy(clf_saved_nb, test_set))
print('svc:',nltk.classify.accuracy(clf_saved_svc, test_set))
print('max:',nltk.classify.accuracy(clf_saved_max, test_set))
'''