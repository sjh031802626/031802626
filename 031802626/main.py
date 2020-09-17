#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import jieba,codecs
from gensim import  corpora,models,similarities
import jieba.posseg as pseg

# 读取停用表
def read_stop_word(file_path):
    file = file_path
    stopwords = codecs.open(file,'r',encoding='utf8').readlines()
    stopwords = [ w.strip() for w in stopwords ]
    return stopwords

# 构建停用词表
stopwords = read_stop_word("stop_word.txt")
# 结巴分词后的停用词性 [标点符号、连词、助词、副词、介词、时语素、‘的’、数词、方位词、代词]
stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

# 创建文件
def create_file(file_path,msg):
    f=open(file_path,'a')
    f.write(msg)
    f.close

# 分词
def cut_words(file):
    with open(file, 'r',encoding="utf-8") as f:
        text = f.read()
        words = jieba.lcut(text,cut_all = True)
    return words

# 去停用词
def drop_Disable_Words(cut_res,stopwords):
    res = []
    for word in cut_res:
        if word in stopwords or word =="\n" or word =="\u3000":
            continue
        res.append(word)
    return res

# 对一篇文章分词、去停用词
def tokenization(file):
    result = []
    with open(file, 'r',encoding='utf8') as f:
        text = f.read()
        words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

def main_gensim(orig_path,copy_path,ans_path):
    files = [orig_path,copy_path]

    corpus = []
    for file in files:
        #分词
        cut_res = cut_words(file)
        #去停用词
        res = drop_Disable_Words(cut_res,stopwords)
        corpus.append(res)

    #建立词袋模型
    dictionary = corpora.Dictionary(corpus)
    txt_vectors = [dictionary.doc2bow(text) for text in corpus]
    
    query = tokenization(orig_path)
    query_bow = dictionary.doc2bow(query)
    TF_IDF(txt_vectors,query_bow)
    
    #建立TF-IDF模型
def TF_IDF(txt_vectors,query_bow):
    tfidf = models.TfidfModel(txt_vectors)
    tfidf_vectors = tfidf[txt_vectors]
    #使用TF-IDF模型计算相似度
    try:
        index = similarities.MatrixSimilarity(tfidf_vectors)
        sims = index[query_bow]
        ans = 1-float(list(enumerate(sims))[0][1])
        create_file(ans_path,copy_path+str('：%.2f\n' % ans))
    except:
        create_file(ans_path,copy_path+"：1.00")
    print(0)
    



orig_path = input('输入原文文件路径：')
copy_path = input('输入相似文本文件路径：')
ans_path = input('答案文件路径：')
main_gensim(orig_path,copy_path,ans_path)
