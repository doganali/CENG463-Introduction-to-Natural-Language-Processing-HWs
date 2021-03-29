
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:35:07 2021

@author: cedgn
"""
import nltk
import sklearn_crfsuite
import sklearn
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import random

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
        'word[:2]': word[-2:],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'upos':upos
        }

    '''if i > 0:
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
        features['EOS'] = True''' 
    
    return features
   
    

def feature_extraction_chunk(sent,i):
    word = sent[i][0]
    pos = sent[i][1]
    features = {
        'bias' : 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[:2]': word[-2:],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'pos':pos
        }
    '''if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2]
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2]
        })
    else:
        features['EOS'] = True '''

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
        #'pos':pos,
        #'chunk':chunk
        }

    '''if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
            #'pos-1': sent[i-1][1]
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
            #'pos+1': sent[i-1][1]
        })
    else:
        features['EOS'] = True '''
      
    return features



def sent2features_pos(sent):
    return [feature_extraction_pos(sent, i) for i in range(len(sent))]
def sent2features_chunk(sent):
    return [feature_extraction_chunk(sent, i) for i in range(len(sent))]
def sent2features_ner(sent):
    return [feature_extraction_ner(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2labels_ner(sent):
    return [label for token, postag,chunk, label in sent]



## POS ##
data_pos = get_data('data/pos/en-ud-train.conllu')
random.shuffle(data_pos)
train_pos, val_pos, test_pos = data_pos[:10000], data_pos[10000:], get_data('data/pos/en-ud-dev.conllu')
X_train = [sent2features_pos(s) for s in train_pos]
y_train = [sent2labels(s) for s in train_pos]
X_val = [sent2features_pos(s) for s in val_pos]
y_val = [sent2labels(s) for s in val_pos]
X_test = [sent2features_pos(s) for s in test_pos]
y_test = [sent2labels(s) for s in test_pos]

crf_pos = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)
crf_pos.fit(X_train, y_train)

y_pred = crf_pos.predict(X_val)
#print(metrics.flat_classification_report(y_val, y_pred))

y_pred = crf_pos.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred),'pos')
cm_pos = nltk.ConfusionMatrix([item for sublist in y_test for item in sublist], [item for sublist in y_pred for item in sublist])
print(cm_pos)

## CHUNKING ##
data_chunk = get_data('data/chunk/train.txt')
random.shuffle(data_chunk)
train_chunk,test_chunk = data_chunk[:7000],data_chunk[7000:]
X_train_chunk = [sent2features_chunk(s) for s in train_chunk]
y_train_chunk = [sent2labels(s) for s in train_chunk]
X_test_chunk = [sent2features_chunk(s) for s in test_chunk]
y_test_chunk = [sent2labels(s) for s in test_chunk]

crf_chunk = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)
crf_chunk.fit(X_train_chunk,y_train_chunk)

y_pred_chunk = crf_chunk.predict(X_test_chunk)
print(metrics.flat_classification_report(y_test_chunk, y_pred_chunk),'chunk')
cm_chunk = nltk.ConfusionMatrix([item for sublist in y_test_chunk for item in sublist], [item for sublist in y_pred_chunk for item in sublist])
print(cm_chunk)
## NER ##
data_ner = get_data('data/ner/eng.train.txt',n_field=4)
random.shuffle(data_ner)
train_ner, val_ner, test_ner = data_ner[:10000] , data_ner[10000:], get_data('data/ner/eng.testa.txt',n_field=4)
X_train_ner = [sent2features_ner(s) for s in train_ner]
y_train_ner = [sent2labels_ner(s) for s in train_ner]
X_val_ner = [sent2features_ner(s) for s in val_ner]
y_val_ner = [sent2labels_ner(s) for s in val_ner]
X_test_ner = [sent2features_ner(s) for s in test_ner]
y_test_ner = [sent2labels_ner(s) for s in test_ner]

crf_ner = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf_ner.fit(X_train_ner,y_train_ner)


y_pred_ner = crf_ner.predict(X_val_ner)
#print(metrics.flat_classification_report(y_val_ner, y_pred_ner),'ner')
cm_ner_val = nltk.ConfusionMatrix([item for sublist in y_val_ner for item in sublist], [item for sublist in y_pred_ner for item in sublist])


y_pred_ner_test = crf_ner.predict(X_test_ner)
print(metrics.flat_classification_report(y_test_ner, y_pred_ner_test),'ner')
cm_ner_test = nltk.ConfusionMatrix([item for sublist in y_test_ner for item in sublist], [item for sublist in y_pred_ner_test for item in sublist])


labels = list(crf_ner.classes_)
labels.remove('O')
print(metrics.flat_classification_report(y_test_ner, y_pred_ner_test,labels = labels))

print(cm_ner_test)