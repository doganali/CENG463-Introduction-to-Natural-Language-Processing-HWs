#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:35:07 2021

@author: cedgn
"""
import time
start_time = time.time()

import nltk
import sklearn
import random
import numpy as np
def get_data(filename,n_field = 3, split_sentences = True):
    
    train_sents = []
    sentence = []
    
    if(n_field== 3):
        for line in open(filename):
            line = line.rstrip()
            if line:
                r = line.split()
                word,f1,f2 = r  # f1:upos,f2:pos is for pos tagging, f1:pos,f2:chunk for chunk tagging
                sentence.append((word,f1,f2))
                
            elif split_sentences:
                train_sents.append(sentence)
                sentence = []
                
        if not split_sentences:
            train_sents = sentence
   
    elif(n_field == 4):
        for line in open(filename):
            line = line.rstrip()
            if line:
                r = line.split()
                word,pos,chunk,ner = r 
                sentence.append((word,pos,chunk,ner))
                
            elif split_sentences:
                train_sents.append(sentence)
                sentence = []
                
        if not split_sentences:
            train_sents = sentence

    
    return train_sents
    
def feature_extraction_pos(sent,i):
    word = sent[i][0]
    upos = sent[i][1]
    features = {
        'bias' : 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[:2]': word[:2],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'upos':upos
        }

    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True
   
    return features

def feature_extraction_chunk(sent,i):
    word = sent[i][0]
    pos = sent[i][1]
    features = {
        'bias' : 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[:2]': word[:2],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos':pos
        }

    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True 
    return features

def feature_extraction_ner(sent,i):
    word = sent[i][0]
    pos = sent[i][1]
    chunk = sent[i][2]
    features = {
        'bias' : 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[:2]': word[:2],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos':pos,
        'chunk':chunk,
        }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True 
   
    return features

def data2features(data,feature_func):
    feature_set = []
    for sent in data:
        for i in range(len(sent)):
            feature_set.append((feature_func(sent,i),sent[i][-1]))
    return feature_set

def obtain_cm(clf,feature_set):
    X = [d[0] for d in feature_set]
    y = [d[1] for d in feature_set]
    preds = clf.classify_many(X)
    cm = nltk.ConfusionMatrix(y,preds)
    clf_results = sklearn.metrics.classification_report(y, preds)
    return cm,clf_results

data_pos = get_data('data/pos/en-ud-train.conllu')
#random.shuffle(data_pos)
train_pos, val_pos, test_pos = data_pos[:5000], data_pos[10000:], get_data('data/pos/en-ud-dev.conllu')

data_chunk = get_data('data/chunk/train.txt')
random.shuffle(data_chunk)
train_chunk,test_chunk = data_chunk[:6000],data_chunk[6000:]


data_ner = get_data('data/ner/eng.train.txt',n_field=4)
random.shuffle(data_ner)
train_ner, val_ner, test_ner = data_ner[:10000], data_ner[10000:], get_data('data/ner/eng.testa.txt',n_field=4)


#POS TAGGING TRAINING
pos_feature_set = data2features(train_pos,feature_extraction_pos)
clf_pos = nltk.MaxentClassifier.train(pos_feature_set,max_iter = 8)


# CHUNKING TRAINING
chunk_feature_set = data2features(train_chunk, feature_extraction_chunk)
clf_chunk = nltk.MaxentClassifier.train(chunk_feature_set,max_iter=5)

# NER TRAINING
ner_feature_set = data2features(train_ner,feature_extraction_ner)
clf_ner = nltk.MaxentClassifier.train(ner_feature_set,max_iter=8)


#POS TAGGING TEST
acc_pos_val = nltk.classify.accuracy(clf_pos, data2features(val_pos, feature_extraction_pos)) 
print('pos val accuracy : ', acc_pos_val) 

acc_pos_test = nltk.classify.accuracy(clf_pos, data2features(test_pos, feature_extraction_pos)) 
print('pos test accuracy : ', acc_pos_test)
words=[]
for sent in train_pos:
    for w in sent:
        if w[0].lower() not in words:
            words.append(w[0].lower())
            
pos_test = data2features(test_pos, feature_extraction_pos)
pos_test_X = [d[0] for d in pos_test]
pos_test_Y = [d[1] for d in pos_test]
preds_pos = clf_pos.classify_many(pos_test_X)
correct = 0 
total = 0 
for i in range(len(preds_pos)):
    if pos_test_X[i]['word.lower()'] not in words:
        total +=1
        if preds_pos[i] == pos_test_Y[i]:
            correct +=1

print('unknown words correct/total : ',correct,'/',total,':',correct/total)

cm_pos_test,cr_pos_test = obtain_cm(clf_pos, data2features(test_pos, feature_extraction_pos))
                        
# CHUNKING test
pos_f_sets_for_chunking = data2features(test_chunk,feature_extraction_pos)
chunk_feature_set_test = data2features(test_chunk,feature_extraction_chunk)
X_pos = [d[0] for d in pos_f_sets_for_chunking]
predicted_pos_tags = clf_pos.classify_many(X_pos)


acc_chunk_test_cheating = nltk.classify.accuracy(clf_chunk,chunk_feature_set_test) # using the pos tags in the file
print('chunking test accuracy with using ground truth pos tags:',acc_chunk_test_cheating)
cm_chunk_test_gt,cr_chunk_test_gt = obtain_cm(clf_chunk,chunk_feature_set_test )

for i,(features,label) in enumerate(chunk_feature_set_test):
    features['pos'] = predicted_pos_tags[i]
acc_chunk_test = nltk.classify.accuracy(clf_chunk,chunk_feature_set_test) # without using the pos tags in the file, instead predicting them
print('chunking test accuracy with using predicted pos tags:',acc_chunk_test)
cm_chunk_test_pt,cr_chunk_test_pt = obtain_cm(clf_chunk,chunk_feature_set_test )

for i,(features,label) in enumerate(chunk_feature_set_test):
    del features['pos']
acc_chunk_test_wo_pos = nltk.classify.accuracy(clf_chunk,chunk_feature_set_test)
print('chunking test accuracy without using any pos tags:',acc_chunk_test_wo_pos)
cm_chunk_test_no_pos,cr_chunk_test_no_pos = obtain_cm(clf_chunk,chunk_feature_set_test )


# NER test
pos_f_set_ner = data2features(test_ner, feature_extraction_pos)
chunk_f_set_ner = data2features(test_ner,feature_extraction_chunk)
ner_feature_set_val = data2features(test_ner,feature_extraction_ner)

X_pos = [d[0] for d in pos_f_set_ner]
X_chunk = [d[0] for d in chunk_f_set_ner]

predicted_pos_tags = clf_pos.classify_many(X_pos)
predicted_chunking_tags = clf_chunk.classify_many(X_chunk)

acc_ner_val_cheating = nltk.classify.accuracy(clf_ner,ner_feature_set_val) # using the pos tags in the file
print('ner test accuracy with using ground truth pos and chunking tags:',acc_ner_val_cheating)
cm_ner_val_gt,cr_ner_val_gt = obtain_cm(clf_ner,ner_feature_set_val)


y_pred =clf_ner.classify_many([d[0] for d in ner_feature_set_val])
labels = list(set(y_pred))
labels.remove('O')
a=sklearn.metrics.classification_report([d[1] for d in ner_feature_set_val], y_pred, labels=labels)

for i,(features,label) in enumerate(ner_feature_set_val):
    features['pos'] = predicted_pos_tags[i]
    features['chunk'] = predicted_chunking_tags[i]
    
acc_ner_val = nltk.classify.accuracy(clf_ner,ner_feature_set_val) # using the pos tags in the file
print('ner test accuracy with using predicted pos and chunking tags:',acc_ner_val)
cm_ner_val_pt,cr_ner_val_pt = obtain_cm(clf_ner,ner_feature_set_val)

for i,(features,label) in enumerate(ner_feature_set_val):
    del features['pos'] 
    del features['chunk'] 
    
acc_ner_val_wo_tags = nltk.classify.accuracy(clf_ner,ner_feature_set_val) # using the pos tags in the file
print('ner test accuracy without using any pos and chunking tags:',acc_ner_val_wo_tags)
cm_ner_val_no_pos,cr_ner_val_no_pos = obtain_cm(clf_ner,ner_feature_set_val)


acc_ner_test = nltk.classify.accuracy(clf_ner,data2features(test_ner,feature_extraction_ner)) # using the pos tags in the file
print('ner test accuracy with using predicted pos and chunking tags:',acc_ner_test)

clf_pos.show_most_informative_features(15)
clf_chunk.show_most_informative_features(15)
clf_ner.show_most_informative_features(15)


print(cr_pos_test)
print(cr_chunk_test_gt)
print(cr_ner_val_gt)


total_time = time.time() - start_time 
print("--- %s seconds ---" % (time.time() - start_time))
