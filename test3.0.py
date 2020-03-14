# coding: UTF-8 -*-
import json
import jieba
import jieba.analyse
import os
import sys
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.externals import joblib
import difflib

def transformtest():
    filepath = r'test_data_sample.json'
    with open(filepath, 'r', encoding='utf-8') as f:
        test_data = []
        item_id = []
        test_data_sample = json.load(f)
        for i in test_data_sample:
            s = str(i['question'] + '\t')
            for k in i['passages']:
                w = s + str(k['content'] + '\n')
                id = str(k['passage_id'])+ '\n'
                test_data.append(w)
                item_id.append(id)
    test_data.pop()
    item_id.pop()
    with open ('item_id.txt','w',encoding='utf-8') as f1:
        for i in item_id:
            f1.write(i)
    return test_data


def stopwordslist(filepath):
    filepath = r'1893（utf8）.txt'
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  # strip()去除首位空白符
    return stopwords

# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence, cut_all=False)  # 精确模式分词
    stopwords = stopwordslist('1893（utf8）.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

def data_jieba():
    data = transformtest()
    savepath = r'testdatajie.txt'
    with open(savepath, 'w', encoding='utf-8') as f:
        for i in data:
            line_seg = seg_sentence(i)  # 这里的返回值是字符串
            f.write(line_seg)


def important():
        with open('testdatajie.txt', 'r', encoding='utf-8') as f1:
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
        return X



def test(X):
    x = np.array(X)
    result = []

    # 读取sample标签
    with open ('item_id.txt','r',encoding='utf-8') as f:
        item_id = []
        n = 0

        for line in f.readlines():
            item_id.append(line[0:8])
            n = n + 1

    bclf = joblib.load('Bayes1.pkl')
    result1 = bclf.predict(x)

    bclf = joblib.load('Bayes2.pkl')
    result2 = bclf.predict(x)

    tclf = joblib.load('tree3.pkl')
    result3 = tclf.predict(x)

    bclf = joblib.load('Bayes4.pkl')
    result4 = bclf.predict(x)

    tclf = joblib.load('tree5.pkl')
    result5 = tclf.predict(x)


    for p in range(len(result1)):
        reresult = result1[p] + result2[p] + result3[p] + result4[p] + result5[p]
        if reresult >= 3:
            q = 1
        else:
            q = 0
        result.append(q)
    return result,item_id,n

if __name__ == '__main__':
    data_jieba()
    X = important()
    test(X)

    lab = test(X)[0]
    item_id = test(X)[1]
    n = test(X)[2]
    print(lab)
    # item_id = transformtest()[1]
    # n = 0
    # for i in item_id:
    #     n = n + 1

    with open('submit.txt','w',encoding='utf-8') as f:
        for i in range(0,n):
            s = str(item_id[i]) + ',' + str(lab[i])
            # s = str(lab[i])
            f.write(s)
            f.write('\n')