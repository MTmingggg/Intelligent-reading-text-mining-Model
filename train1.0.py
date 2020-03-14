# coding: UTF-8 -*-
import json
import jieba
import jieba.analyse
import os
import sys
import numpy as np
import re
import random
from sklearn.naive_bayes import GaussianNB
from sklearn import naive_bayes,tree
from sklearn import ensemble
from sklearn.externals import joblib
import difflib
from sklearn import naive_bayes,svm


def divideandmodel():

    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    Y5 = []

    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []

    with open('lab.txt','r',encoding='utf-8') as f:
        for i in f.readlines():
            Y = eval(i)
    with open('datajie.txt','r',encoding='utf-8') as f1:
        X = []
        for line in f1:
            keywords = jieba.analyse.extract_tags(line, topK=8, withWeight=True)
            if (len(keywords) != 0):
                s1 = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
                index1 = 0
                for item in keywords:
                    s1[index1] = '%.6f' % item[1]
                    index1 = index1 + 1
            else:
                s1 = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
            X.append(list(map(float, (s1))))


    for i in range(int((len(Y)*4)/5)):
        a = random.randint(0,len(Y)-1)
        X1.append(X[a])
        Y1.append(Y[a])

        b = random.randint(0,len(Y)-1)
        X2.append(X[b])
        Y2.append(Y[b])

        c = random.randint(0,len(Y)-1)
        X3.append(X[c])
        Y3.append(Y[c])

        d = random.randint(0,len(Y)-1)
        X4.append(X[d])
        Y4.append(Y[d])

        e = random.randint(0, len(Y)- 1)
        X5.append(X[e])
        Y5.append(Y[e])


    x = np.array(X1)
    y = np.array(Y1)

    treeclf = tree.DecisionTreeClassifier(criterion='gini')
    treeclf.fit(x, y)
    joblib.dump(treeclf, 'tree1.pkl')
    bayesclf = naive_bayes.GaussianNB()
    bayesclf.fit(x, y)
    joblib.dump(bayesclf, 'bayes1.pkl')

    x = np.array(X2)
    y = np.array(Y2)
    treeclf.fit(x, y)
    joblib.dump(treeclf, 'tree2.pkl')
    bayesclf.fit(x, y)
    joblib.dump(bayesclf, 'bayes2.pkl')

    x = np.array(X3)
    y = np.array(Y3)
    treeclf.fit(x, y)
    joblib.dump(treeclf, 'tree3.pkl')
    bayesclf.fit(x, y)
    joblib.dump(bayesclf, 'bayes3.pkl')

    x = np.array(X4)
    y = np.array(Y4)
    treeclf.fit(x, y)
    joblib.dump(treeclf, 'tree4.pkl')
    bayesclf.fit(x, y)
    joblib.dump(bayesclf, 'bayes4.pkl')

    x = np.array(X5)
    y = np.array(Y5)
    treeclf.fit(x, y)
    joblib.dump(treeclf, 'tree5.pkl')
    bayesclf.fit(x, y)
    joblib.dump(bayesclf, 'bayes5.pkl')

if __name__=='__main__':
    divideandmodel()
