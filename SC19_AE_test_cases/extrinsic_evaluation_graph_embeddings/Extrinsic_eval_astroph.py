#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:31:07 2019

@author: moon.310
"""


import csv
import codecs
import array
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import random
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC


def _readTxt(fname, size_only=False, first_n=None, filter_to=None, lower_keys=False,
        errors='strict', separator=' ', skip_parsing_errors=False):
    '''Returns array of words and word embedding matrix
    '''
    words, vectors = [], []
    hook = open(fname, 'rb')

    bsep = bytes(separator, 'utf-8')[0]

    if filter_to:
        if lower_keys:
            filter_set = set([f.lower() for f in filter_to])
            key_filter = lambda k: k.lower() in filter_set
        else:
            filter_set = set(filter_to)
            key_filter = lambda k: k in filter_set
    else:
        key_filter = lambda k: True

    # get summary info about vectors file
    (numWords, dim) = (int(s.strip()) for s in hook.readline().decode('utf-8', errors=errors).split())
    if size_only:
        return (numWords, dim)

    line_ix = 0
    for line in hook:
        line_ix += 1
        if len(line.strip()) > 0:
            try:
                ix = 0
                while line[ix] != bsep:
                    ix += 1
                key = line[:ix].decode('utf-8', errors=errors)
                vector = line[ix+1:].decode('utf-8', errors=errors)
                vector = np.array([float(n) for n in vector.split()])
            except Exception as e:
                if skip_parsing_errors:
                    sys.stderr.write('[WARNING] Line %d: Parsing error -- %s\n' % (line_ix, str(e)))
                    line_ix += 1
                    continue
                else:
                    print("<<< CHUNKING ERROR >>>")
                    print(line)
                    raise e

            #chunks = line.split()
            #try:
            #    key, vector = chunks[0].strip(), np.array([float(n) for n in chunks[1:]])
            if len(vector) == dim - 1 or len(key) == 0:
                sys.stderr.write("[WARNING] Line %d: Read vector without a key, skipping\n" % line_ix)
            elif len(vector) != dim:
                raise ValueError("Read %d-length vector, expected %d" % (len(vector), dim))
            else:
                if key_filter(key):
                    words.append(key)
                    vectors.append(vector)

                if (not first_n is None) and len(words) == first_n:
                    break
            line_ix += 1
    hook.close()

    if not first_n is None:
        assert len(words) == first_n
    elif not filter_to:
        if len(words) != numWords:
            sys.stderr.write("[WARNING] Expected %d words, read %d\n" % (numWords, len(words)))

    return (words, vectors)

edges_array = []
with open('CA-AstroPh_edges.txt', 'r') as f:
    for line in f:
        edges_array.append(line.split())
        
        
output_word2vec_cpu = _readTxt('../word2vec_cpu_vectors_astroph.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_pWord2Vec_cpu = _readTxt('../pWord2Vec_cpu_vectors_astroph.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_wombatSGNS_cpu = _readTxt('../wombatSGNS_cpu_vectors.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_pSGNScc_cpu = _readTxt('../pSGNScc_cpu_vectors_astroph.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_PAR_Word2Vec_cpu = _readTxt('../PAR_Word2Vec_cpu_vectors_astroph.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_accSGNS_gpu = _readTxt('../accSGNS_gpu_vectors_astroph.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_PAR_Word2Vec_gpu = _readTxt('../PAR_Word2Vec_gpu_vectors_astroph.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)


wordId = output_word2vec_cpu[0] # vocabulary size, number of unique nodes
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_word2vec_cpu[1] # vacabulary size * hidden size
del word_embed[0]

        
X_positive = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_positive = np.zeros(len(edges_array))

for i in range(len(edges_array)):
    X_positive[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(edges_array[i][0]))]
    X_positive[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(edges_array[i][1]))]
    y_positive[i] = 1
    

X_negative = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_negative = np.zeros(len(edges_array))


random_pair_array = []
    
random.seed(614)
i = 0;
while i < len(edges_array):
    random_pair = [str(wordId_int[random.randint(0,len(wordId_int)-1)]), str(wordId_int[random.randint(0,len(wordId_int)-1)])]
    if not any(random_pair == x for x in edges_array):
            random_pair_array.append(random_pair)
            X_negative[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(random_pair[0]))]
            X_negative[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(random_pair[1]))]
            y_negative[i] = 0
            i += 1
        
X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)
X, y = shuffle(X, y, random_state=17)

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=82, tol=1e-05, verbose=0)
Word2Vec_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')



wordId = output_pWord2Vec_cpu[0]
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_pWord2Vec_cpu[1]
del word_embed[0]
        

X_positive = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_positive = np.zeros(len(edges_array))

for i in range(len(edges_array)):
    X_positive[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(edges_array[i][0]))]
    X_positive[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(edges_array[i][1]))]
    y_positive[i] = 1
    

X_negative = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_negative = np.zeros(len(edges_array))


for i in range(len(random_pair_array)):
    X_negative[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(random_pair_array[i][0]))]
    X_negative[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(random_pair_array[i][1]))]
    y_negative[i] = 0
    

X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)
X, y = shuffle(X, y, random_state=17)

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=82, tol=1e-05, verbose=0)
pWord2Vec_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')



wordId = output_wombatSGNS_cpu[0]
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_wombatSGNS_cpu[1]
del word_embed[0]
        

X_positive = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_positive = np.zeros(len(edges_array))

for i in range(len(edges_array)):
    X_positive[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(edges_array[i][0]))]
    X_positive[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(edges_array[i][1]))]
    y_positive[i] = 1
    

X_negative = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_negative = np.zeros(len(edges_array))


for i in range(len(random_pair_array)):
    X_negative[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(random_pair_array[i][0]))]
    X_negative[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(random_pair_array[i][1]))]
    y_negative[i] = 0
    

X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)
X, y = shuffle(X, y, random_state=17)

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=82, tol=1e-05, verbose=0)
wombatSGNS_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')




wordId = output_pSGNScc_cpu[0]
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_pSGNScc_cpu[1]
del word_embed[0]
        

X_positive = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_positive = np.zeros(len(edges_array))

for i in range(len(edges_array)):
    X_positive[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(edges_array[i][0]))]
    X_positive[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(edges_array[i][1]))]
    y_positive[i] = 1
    

X_negative = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_negative = np.zeros(len(edges_array))


for i in range(len(random_pair_array)):
    X_negative[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(random_pair_array[i][0]))]
    X_negative[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(random_pair_array[i][1]))]
    y_negative[i] = 0
    

X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)
X, y = shuffle(X, y, random_state=17)

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=82, tol=1e-05, verbose=0)
pSGNScc_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')




wordId = output_PAR_Word2Vec_cpu[0]
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_PAR_Word2Vec_cpu[1]
del word_embed[0]
        

X_positive = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_positive = np.zeros(len(edges_array))

for i in range(len(edges_array)):
    X_positive[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(edges_array[i][0]))]
    X_positive[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(edges_array[i][1]))]
    y_positive[i] = 1
    

X_negative = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_negative = np.zeros(len(edges_array))


for i in range(len(random_pair_array)):
    X_negative[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(random_pair_array[i][0]))]
    X_negative[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(random_pair_array[i][1]))]
    y_negative[i] = 0
    

X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)
X, y = shuffle(X, y, random_state=17)

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=82, tol=1e-05, verbose=0)
PAR_Word2Vec_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')



wordId = output_accSGNS_gpu[0]
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_accSGNS_gpu[1]
del word_embed[0]
        

X_positive = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_positive = np.zeros(len(edges_array))

for i in range(len(edges_array)):
    X_positive[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(edges_array[i][0]))]
    X_positive[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(edges_array[i][1]))]
    y_positive[i] = 1
    

X_negative = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_negative = np.zeros(len(edges_array))


for i in range(len(random_pair_array)):
    X_negative[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(random_pair_array[i][0]))]
    X_negative[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(random_pair_array[i][1]))]
    y_negative[i] = 0
    

X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)
X, y = shuffle(X, y, random_state=17)

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=82, tol=1e-05, verbose=0)
accSGNS_gpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')



wordId = output_PAR_Word2Vec_gpu[0]
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_PAR_Word2Vec_gpu[1]
del word_embed[0]
        

X_positive = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_positive = np.zeros(len(edges_array))

for i in range(len(edges_array)):
    X_positive[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(edges_array[i][0]))]
    X_positive[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(edges_array[i][1]))]
    y_positive[i] = 1
    

X_negative = np.zeros(( len(edges_array), len(word_embed[0])*2 ))
y_negative = np.zeros(len(edges_array))


for i in range(len(random_pair_array)):
    X_negative[i,0:len(word_embed[0])] = word_embed[wordId_int.index(int(random_pair_array[i][0]))]
    X_negative[i,len(word_embed[0]):len(word_embed[0])*2] = word_embed[wordId_int.index(int(random_pair_array[i][1]))]
    y_negative[i] = 0
    

X = np.concatenate((X_positive, X_negative), axis=0)
y = np.concatenate((y_positive, y_negative), axis=0)
X, y = shuffle(X, y, random_state=17)

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=82, tol=1e-05, verbose=0)
PAR_Word2Vec_gpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')

print("------------------------------------------------------------")
print("ASTRO-PH dataset: Extrinsic evaluation results")
print("------------------------------------------------------------")


print("ASTRO-PH - Word2Vec-cpu micro F1 score: %0.4f" % (Word2Vec_cpu_micro_scores.mean()))

print("ASTRO-PH - pWord2Vec-cpu micro F1 score: %0.4f" % (pWord2Vec_cpu_micro_scores.mean()))

print("ASTRO-PH - wombatSGNS-cpu micro F1 score: %0.4f" % (wombatSGNS_cpu_micro_scores.mean()))

print("ASTRO-PH - pSGNScc-cpu micro F1 score: %0.4f" % (pSGNScc_cpu_micro_scores.mean()))

print("ASTRO-PH - PAR-Word2Vec-cpu micro F1 score: %0.4f" % (PAR_Word2Vec_cpu_micro_scores.mean()))

print("ASTRO-PH - accSGNS-gpu micro F1 score: %0.4f" % (accSGNS_gpu_micro_scores.mean()))

print("ASTRO-PH - PAR-Word2Vec-gpu micro F1 score: %0.4f" % (PAR_Word2Vec_gpu_micro_scores.mean()))
