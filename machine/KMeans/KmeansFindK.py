# !/usr/bin/python
# coding:utf-8
"""
无监督学习，KMeans,寻找最优的K
训练数据

@ author: Pruce
@ email:    1756983926@qq.com
@ date:     Created on Feb 16, 2018
https://blog.csdn.net/qq_15738501/article/details/79036255
"""

import pandas as pd  
from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_score  
import matplotlib.pyplot as plt  
import gc

"""
df_features = pd.read_csv('D:\class2_active_3.txt',encoding='utf8')  
df_feature = df_features[['class1id_4001_class2rating',
                               'class1id_4001_gamecount',
                               'class1id_4001_onlineday',
                               'class1id_4001_onlinetime',
                               'class1id_4001_lostgamecount']].sample(n=50000)
df_feature.to_csv('D:\class2_active_3_sample.txt', encoding='utf-8', sep=',')
"""

df_features = pd.read_csv('D:\class2_active_3_sample.txt',encoding='utf8')  
df_feature = df_features[['class1id_4001_class2rating',
                               'class1id_4001_gamecount',
                               'class1id_4001_onlineday',
                               'class1id_4001_onlinetime',
                               'class1id_4001_lostgamecount']]

Scores = []  # 存放轮廓系数  
X = range(2,25)
for k in X:      
    estimator = KMeans(n_clusters=k, max_iter=500).fit(df_feature)  # 构造聚类器  
    Scores.append(silhouette_score(df_feature, estimator.labels_, metric='euclidean'))  
    
    print(k, 'kmeans analysis ', estimator.inertia_)
    frequency = {}
    for ilabel in estimator.labels_:
        if ilabel not in frequency:
            frequency[ilabel] = 1
        else:
            frequency[ilabel] += 1
    print(frequency)
    
    del estimator
    gc.collect()
    
plt.xlabel('k')  
plt.ylabel('轮廓系数')  
plt.plot(X, Scores, 'o-')  
plt.show()  