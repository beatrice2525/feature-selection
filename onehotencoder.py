# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:11:04 2018

@author: beatr
"""

#热编码
#变量只有一个，例如，三分类
from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [{'city': 'New York'},{'city': 'San Francisco'}, {'city': 'Chapel Hill'}]
print(onehot_encoder.fit_transform(instances).toarray())

#多个变量，每个变量多分雷，给出一个新的值得到分类编码
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()

enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

enc.transform([[0, 1, 3]]).toarray()