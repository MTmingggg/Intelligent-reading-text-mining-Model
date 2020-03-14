# coding: UTF-8 -*-

import json
import os
import sys
import re
import jieba


def transform():
    filepath = r'train_data_complete.json'
    with open(filepath, 'r', encoding='utf-8') as f:
        lab = []
        train_data = []
        train_data_sample = json.load(f)
        # f1 = open(savepath, 'w', encoding='utf-8')
        for i in train_data_sample:
            s = str(i['question'] + '\t')
        # s = str(i['item_id']) + '\t' + i['question'] + '\t'
            for k in i['passages']:
                w = s + str(k['content'] + '\n')
                w1 = k["label"]
                train_data.append(w)
                lab.append(w1)
        with open('lab.txt','w',encoding='utf-8') as f1:
            f1.write(str(lab))
        train_data.pop()
        return train_data


# 创建停用词list

def stopwordslist(filepath):
    filepath = r'1893（utf8）.txt'
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()] #strip()去除首位空白符
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
    data = transform()
    savepath = r'datajie.txt'
    with open(savepath,'w',encoding='utf-8') as f:
        for i in data:
            line_seg = seg_sentence(i) # 这里的返回值是字符串
            f.write(line_seg)

if __name__ == '__main__':
    data_jieba()