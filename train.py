# coding: UTF-8 -*-
import jieba.analyse
import os
import numpy as np
import sys
import re
from sklearn.naive_bayes import GaussianNB
from sklearn import naive_bayes,svm,tree
from sklearn import ensemble
from sklearn.externals import joblib
import difflib


def important():
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
    return X,Y


def model(X,Y):
    x = np.array(X)
    y = np.array(Y)

    Forestclf = ensemble.RandomForestClassifier(n_estimators=50,)
    Forestclf.fit(x, y)
    joblib.dump(Forestclf, 'Forest.pkl')
    fclf = joblib.load('Forest.pkl')
    #
    # bayesclf = naive_bayes.GaussianNB()
    # bayesclf.fit(x, y)
    # joblib.dump(bayesclf, 'bayes.pkl')
    # Bclf = joblib.load('bayes.pkl')
    #
    # SVCclf = svm.SVC()
    # SVCclf.fit(x, y)
    # joblib.dump(SVCclf, 'SVC.pkl')
    # Sclf = joblib.load('SVC.pkl')
    #
    # treeclf = tree.DecisionTreeClassifier(criterion='gini')
    # treeclf.fit(x, y)
    # joblib.dump(treeclf, 'tree.pkl')
    # Tclf = joblib.load('tree.pkl')

if __name__ == '__main__':
    X,Y = important()
    model(X,Y)