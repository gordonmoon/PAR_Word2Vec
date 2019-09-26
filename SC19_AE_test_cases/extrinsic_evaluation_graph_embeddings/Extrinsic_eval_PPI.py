#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import codecs
import array
import sys
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
    

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

mat_contents = scipy.io.loadmat('Homo_sapiens.mat')

gp = mat_contents['group']
group_array = gp.toarray()

nodeId = [];
nodeId_label = [];

for r in range(group_array.shape[0]):
    for c in range(group_array.shape[1]):
        if (group_array[r][c] == 1):
            nodeId.append(r+1)
            nodeId_label.append(c+1)
            
nodeId_int = [int(i) for i in nodeId]
nodeId_label_int = [int(i) for i in nodeId_label]


output_word2vec_cpu = _readTxt('../word2vec_cpu_vectors_ppi.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_pWord2Vec_cpu = _readTxt('../pWord2Vec_cpu_vectors_ppi.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_wombatSGNS_cpu = _readTxt('../wombatSGNS_cpu_vectors.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
# output_pSGNScc_cpu = _readTxt('../pSGNScc_cpu_vectors_ppi.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_PAR_Word2Vec_cpu = _readTxt('../PAR_Word2Vec_cpu_vectors_ppi.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_accSGNS_gpu = _readTxt('../accSGNS_gpu_vectors_ppi.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)
output_PAR_Word2Vec_gpu = _readTxt('../PAR_Word2Vec_gpu_vectors_ppi.txt', size_only=False,first_n=None, filter_to=None, lower_keys=False, errors='strict',separator=' ', skip_parsing_errors=False)

wordId = output_word2vec_cpu[0] # vocabulary size
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_word2vec_cpu[1] # vacabulary size * hidden size
del word_embed[0]

X = np.zeros((len(nodeId_int), len(word_embed[0])))
y = np.zeros(len(nodeId_int))

for i in range(len(nodeId_int)):
    mapId = wordId_int.index(nodeId_int[i])
    X[i,:] = word_embed[mapId]
    y[i] = nodeId_label_int[i]

clf = LogisticRegression(random_state=79, solver='lbfgs',multi_class='multinomial',max_iter=10000)
Word2Vec_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')
Word2Vec_cpu_macro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')



wordId = output_pWord2Vec_cpu[0] # vocabulary size
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_pWord2Vec_cpu[1] # vacabulary size * hidden size
del word_embed[0]

X = np.zeros((len(nodeId_int), len(word_embed[0])))
y = np.zeros(len(nodeId_int))

for i in range(len(nodeId_int)):
    mapId = wordId_int.index(nodeId_int[i])
    X[i,:] = word_embed[mapId]
    y[i] = nodeId_label_int[i]

clf = LogisticRegression(random_state=79, solver='lbfgs',multi_class='multinomial',max_iter=10000)
pWord2Vec_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')
pWord2Vec_cpu_macro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')




wordId = output_wombatSGNS_cpu[0] # vocabulary size
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_wombatSGNS_cpu[1] # vacabulary size * hidden size
del word_embed[0]

X = np.zeros((len(nodeId_int), len(word_embed[0])))
y = np.zeros(len(nodeId_int))

for i in range(len(nodeId_int)):
    mapId = wordId_int.index(nodeId_int[i])
    X[i,:] = word_embed[mapId]
    y[i] = nodeId_label_int[i]

clf = LogisticRegression(random_state=79, solver='lbfgs',multi_class='multinomial',max_iter=10000)
wombatSGNS_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')
wombatSGNS_cpu_macro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')




# wordId = output_pSGNScc_cpu[0] # vocabulary size
# del wordId[0]
# wordId_int = [int(i) for i in wordId]

# word_embed = output_pSGNScc_cpu[1] # vacabulary size * hidden size
# del word_embed[0]

# X = np.zeros((len(nodeId_int), len(word_embed[0])))
# y = np.zeros(len(nodeId_int))

# for i in range(len(nodeId_int)):
#     mapId = wordId_int.index(nodeId_int[i])
#     X[i,:] = word_embed[mapId]
#     y[i] = nodeId_label_int[i]

# clf = LogisticRegression(random_state=79, solver='lbfgs',multi_class='multinomial',max_iter=10000)
# pSGNScc_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')
# pSGNScc_cpu_macro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')




wordId = output_PAR_Word2Vec_cpu[0] # vocabulary size
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_PAR_Word2Vec_cpu[1] # vacabulary size * hidden size
del word_embed[0]

X = np.zeros((len(nodeId_int), len(word_embed[0])))
y = np.zeros(len(nodeId_int))

for i in range(len(nodeId_int)):
    mapId = wordId_int.index(nodeId_int[i])
    X[i,:] = word_embed[mapId]
    y[i] = nodeId_label_int[i]

clf = LogisticRegression(random_state=79, solver='lbfgs',multi_class='multinomial',max_iter=10000)
PAR_Word2Vec_cpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')
PAR_Word2Vec_cpu_macro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')



wordId = output_accSGNS_gpu[0] # vocabulary size
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_accSGNS_gpu[1] # vacabulary size * hidden size
del word_embed[0]

X = np.zeros((len(nodeId_int), len(word_embed[0])))
y = np.zeros(len(nodeId_int))

for i in range(len(nodeId_int)):
    mapId = wordId_int.index(nodeId_int[i])
    X[i,:] = word_embed[mapId]
    y[i] = nodeId_label_int[i]

clf = LogisticRegression(random_state=79, solver='lbfgs',multi_class='multinomial',max_iter=10000)
accSGNS_gpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')
accSGNS_gpu_macro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')



wordId = output_PAR_Word2Vec_gpu[0] # vocabulary size
del wordId[0]
wordId_int = [int(i) for i in wordId]

word_embed = output_PAR_Word2Vec_gpu[1] # vacabulary size * hidden size
del word_embed[0]

X = np.zeros((len(nodeId_int), len(word_embed[0])))
y = np.zeros(len(nodeId_int))

for i in range(len(nodeId_int)):
    mapId = wordId_int.index(nodeId_int[i])
    X[i,:] = word_embed[mapId]
    y[i] = nodeId_label_int[i]

clf = LogisticRegression(random_state=79, solver='lbfgs',multi_class='multinomial',max_iter=10000)
PAR_Word2Vec_gpu_micro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_micro')
PAR_Word2Vec_gpu_macro_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')



print("------------------------------------------------------------")
print("PPI dataset: Extrinsic evaluation results")
print("------------------------------------------------------------")

print("PPI - Word2Vec-cpu micro F1 score: %0.4f macro F1 score: %0.4f" % ((Word2Vec_cpu_micro_scores.mean()), (Word2Vec_cpu_macro_scores.mean())))

print("PPI - pWord2Vec-cpu micro F1 score: %0.4f macro F1 score: %0.4f" % ((pWord2Vec_cpu_micro_scores.mean()), (pWord2Vec_cpu_macro_scores.mean())))

print("PPI - wombatSGNS-cpu micro F1 score: %0.4f macro F1 score: %0.4f" % ((wombatSGNS_cpu_micro_scores.mean()), (wombatSGNS_cpu_macro_scores.mean())))

# print("PPI - pSGNScc-cpu micro F1 score: %0.4f macro F1 score: %0.4f" % ((pSGNScc_cpu_micro_scores.mean()), (pSGNScc_cpu_macro_scores.mean())))

print("PPI - PAR-Word2Vec-cpu micro F1 score: %0.4f macro F1 score: %0.4f" % ((PAR_Word2Vec_cpu_micro_scores.mean()), (PAR_Word2Vec_cpu_macro_scores.mean())))

print("PPI - accSGNS-gpu micro F1 score: %0.4f macro F1 score: %0.4f" % ((accSGNS_gpu_micro_scores.mean()), (accSGNS_gpu_macro_scores.mean())))

print("PPI - PAR-Word2Vec-gpu micro F1 score: %0.4f macro F1 score: %0.4f" % ((PAR_Word2Vec_gpu_micro_scores.mean()), (PAR_Word2Vec_gpu_macro_scores.mean())))
