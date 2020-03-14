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

    # 读取sample标签
    with open ('submit_sample.txt','r',encoding='utf-8') as f:
        Y = []
        item_id = []
        n = 0

        for line in f.readlines():
            Y.append(line[9])
            item_id.append(line[0:8])
            n = n + 1

#计算随机森林的准确率
    fclf = joblib.load('Forest.pkl')
    result1 = fclf.predict(x)
    j = 0
    right1 = 0
    number1 = 0
    for i in result1:
        if (i == int(Y[j])):
            right1 = right1 + 1
            number1 = number1 + 1
        else:
            number1 = number1 + 1
        j = j + 1
    prob1 = right1 / number1

    real1 = 0  # 预测为1且标签也为1的个数
    guss = 0  # 预测为1的个数
    lab1 = 0  # 标签为1的个数

    j = 0
    for i in result1:
        if (i == 1):
            if (int(Y[j]) == 1):
                real1 = real1 + 1
                guss = guss + 1
                lab1 = lab1 + 1
            else:
                guss = guss + 1
        else:
            if (int(Y[j]) == 1):
                lab1 = lab1 + 1
        j = j + 1
    F1 = (real1 * 2) / (guss + lab1)

#计算贝叶斯的准确率
    bclf = joblib.load('Bayes.pkl')
    result2 = bclf.predict(x)
    j = 0
    right2 = 0
    number2 = 0
    for i in result2:
        if (i == int(Y[j])):
            right2 = right2 + 1
            number2 = number2 + 1
        else:
            number2 = number2 + 1
        j = j + 1
    prob2 = right2 / number2

    real1 = 0  # 预测为1且标签也为1的个数
    guss = 0  # 预测为1的个数
    lab1 = 0  # 标签为1的个数

    j = 0
    for i in result2:
        if (i == 1):
            if (int(Y[j]) == 1):
                real1 = real1 + 1
                guss = guss + 1
                lab1 = lab1 + 1
            else:
                guss = guss + 1
        else:
            if (int(Y[j]) == 1):
                lab1 = lab1 + 1
        j = j + 1
    F2 = (real1 * 2) / (guss + lab1)

#计算支持向量机的准确率
    rclf = joblib.load('SVC.pkl')
    result3 = rclf.predict(x)
    j = 0
    right3 = 0
    number3 = 0
    for i in result3:
        if (i == int(Y[j])):
            right3 = right3 + 1
            number3 = number3 + 1
        else:
            number3 = number3 + 1
        j = j + 1
    prob3 = right3 / number3

    real1 = 0  # 预测为1且标签也为1的个数
    guss = 0  # 预测为1的个数
    lab1 = 0  # 标签为1的个数

    j = 0
    for i in result3:
        if (i == 1):
            if (int(Y[j]) == 1):
                real1 = real1 + 1
                guss = guss + 1
                lab1 = lab1 + 1
            else:
                guss = guss + 1
        else:
            if (int(Y[j]) == 1):
                lab1 = lab1 + 1
        j = j + 1
    F3 = (real1 * 2) / (guss + lab1)

# 计算决策树的准确率
    Tclf = joblib.load('tree.pkl')
    result4 = Tclf.predict(x)
    j = 0
    right4 = 0
    number4 = 0
    for i in result4:
        if (i == int(Y[j])):
            right4 = right4 + 1
            number4 = number4 + 1
        else:
            number4 = number4 + 1
        j = j + 1
    prob4 = right4 / number4

    real1 = 0  # 预测为1且标签也为1的个数
    guss = 0  # 预测为1的个数
    lab1 = 0  # 标签为1的个数

    j = 0
    for i in result4:
        if (i == 1):
            if (int(Y[j]) == 1):
                real1 = real1 + 1
                guss = guss + 1
                lab1 = lab1 + 1
            else:
                guss = guss + 1
        else:
            if (int(Y[j]) == 1):
                lab1 = lab1 + 1
        j = j + 1
    F4 = (real1 * 2) / (guss + lab1)

    # j = 0
    # for i in result:
    #     if (i == int(Y[j])):
    #         right = right + 1
    #         number = number + 1
    #     else:
    #         number = number + 1
    #     j = j + 1
    # prob = right / number
    return prob1,prob2,prob3,prob4,F1,F2,F3,F4
    # return result,item_id,n,prob

if __name__ == '__main__':
    data_jieba()
    X = important()

    # test(X)
    # lab = test(X)[0]
    # item_id = test(X)[1]
    # n = test(X)[2]

    print('随机生成树准确率为',test(X)[0])
    print('随机生成树F1得分为', test(X)[4])

    print('朴素贝叶斯准确率为',test(X)[1])
    print('朴素贝叶斯F1得分为', test(X)[5])

    print('支持向量机准确率为',test(X)[2])
    print('支持向量机F1得分为', test(X)[6])

    print('决策树准确率为', test(X)[3])
    print('决策树F1得分为', test(X)[7])

    # with open('submit.txt','w',encoding='utf-8') as f:
    #     for i in range(0,n):
    #         s = str(item_id[i]) + ',' + str(lab[i])
    #         f.write(s)
    #         f.write('\n')