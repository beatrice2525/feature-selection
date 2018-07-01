# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:36:20 2018

@author: beatr
"""

#CountVectorizer类会把文档全部转换成小写，然后将文档词块化（tokenize）。
#文档词块化是把句子分割成词块（token）或有意义的字母序列的过程。词块大多是单词，
#但是他们也可能是一些短语，如标点符号和词缀。CountVectorizer类通过正则表达式用空格分割句子，
#然后抽取长度大于等于2的字母序列。
#缺点，当文档里有a等一个长度单词时无法识别.scikit-learn实现代码如下：
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())#输出向量
print(vectorizer.vocabulary_)#词的索引

#判定文档之间的距离
from sklearn.metrics.pairwise import euclidean_distances
counts = vectorizer.fit_transform(corpus).todense()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))

#words of bags 容易形成稀疏矩阵，需要降维
#1.停用词过滤，stop_words
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

#2.词根还原与词形还原
#特征向量里面的单词很多都是一个词的不同形式，比如jumping和jumps都是jump的不同形式。词根还原与词形还原就是为了将单词从不同的时态、派生形式还原
#词形还原就是用来处理可以表现单词意思的词元（lemma）或形态学的词根（morphological root）的过程，
#词元是单词在词典中查询该词的基本形式。词根还原与词形还原类似，但它不是生成单词的形态学的词根。
#而是把附加的词缀都去掉，构成一个词块，可能不是一个正常的单词。词形还原通常需要词法资料的支持，比如WordNet和单词词类（part of speech）。
#词根还原算法通常需要用规则产生词干（stem）并操作词块，不需要词法资源，也不在乎单词的意思
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]
vectorizer = CountVectorizer(binary=True, stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

corpus = [
    'I am gathering ingredients for the sandwich.',
    'There were many wizards at the gathering.'
]
import nltk
nltk.download()#Python的NLTK（Natural Language Tool Kit）库
#提取词元
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))
