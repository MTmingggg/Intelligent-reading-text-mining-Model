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
        test_data_sample = json.load(f)
        for i in test_data_sample:
            s = str(i['question'] + '\t')
            for k in i['passages']:
                w = s + str(k['content'] + '\n')
                test_data.append(w)
    test_data.pop()
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
    right = 0
    number = 0
    result = []
    # 读取sample标签
    with open ('submit_sample.txt','r',encoding='utf-8') as f:
        Y = []
        item_id = []
        n = 0

        for line in f.readlines():
            Y.append(line[9])
            item_id.append(line[0:8])
            n = n + 1

    # rclf = joblib.load('SVC1.pkl')
    # result1 = rclf.predict(x)
    # fclf = joblib.load('Forest1.pkl')
    # result1 = fclf.predict(x)
    bclf = joblib.load('Bayes1.pkl')
    result1 = bclf.predict(x)
    # tclf = joblib.load('tree1.pkl')
    # result1 = tclf.predict(x)

    # rclf = joblib.load('SVC2.pkl')
    # result2 = rclf.predict(x)
    # fclf = joblib.load('Forest2.pkl')
    # result2 = fclf.predict(x)
    bclf = joblib.load('Bayes2.pkl')
    result2 = bclf.predict(x)
    # tclf = joblib.load('tree2.pkl')
    # result2 = tclf.predict(x)

    # rclf = joblib.load('SVC3.pkl')
    # result3 = rclf.predict(x)
    # fclf = joblib.load('Forest3.pkl')
    # result3 = fclf.predict(x)
    # bclf = joblib.load('Bayes3.pkl')
    # result3 = bclf.predict(x)
    tclf = joblib.load('tree3.pkl')
    result3 = tclf.predict(x)

    # rclf = joblib.load('SVC4.pkl')
    # result4 = rclf.predict(x)
    # fclf = joblib.load('Forest4.pkl')
    # result4 = fclf.predict(x)
    bclf = joblib.load('Bayes4.pkl')
    result4= bclf.predict(x)
    # tclf = joblib.load('tree4.pkl')
    # result4 = tclf.predict(x)

    # rclf = joblib.load('SVC5.pkl')
    # result5 = rclf.predict(x)
    # fclf = joblib.load('Forest5.pkl')
    # result5 = fclf.predict(x)
    # bclf = joblib.load('Bayes5.pkl')
    # result5 = bclf.predict(x)
    tclf = joblib.load('tree5.pkl')
    result5 = tclf.predict(x)


    for p in range(len(result1)):
        reresult = result1[p] + result2[p] + result3[p] + result4[p] + result5[p]
        # reresult = result1[p] + result2[p] + result3[p]+result5[p]
        if reresult >= 3:
            q = 1
        else:
            q = 0
        result.append(q)

    j = 0
    for i in result:
        if (i == int(Y[j])):
            right = right + 1
            number = number + 1
        else:
            number = number + 1
        j = j + 1
    prob = right / number

    real1 = 0 #预测为1且标签也为1的个数
    guss  = 0 #预测为1的个数
    lab1 = 0 #标签为1的个数

    j = 0
    for i in result:
        if (i == 1):
            if (int(Y[j]) == 1):
                real1 = real1 + 1
                guss = guss + 1
                lab1 = lab1 + 1
            else:
                guss =  guss + 1
        else:
            if (int(Y[j]) == 1):
                lab1 = lab1 + 1
        j = j + 1
    F1=(real1*2)/(guss + lab1)

    return result,item_id,n,prob,F1

if __name__ == '__main__':
    data_jieba()
    X = important()
    test(X)

    lab = test(X)[0]
    item_id = test(X)[1]
    n = test(X)[2]

    print('优化后模型准确率为',test(X)[3])
    print('优化后模型F1得分为',test(X)[4])

    with open('submit.txt','w',encoding='utf-8') as f:
        for i in range(0,n):
            s = str(item_id[i]) + ',' + str(lab[i])
            f.write(s)
            f.write('\n')