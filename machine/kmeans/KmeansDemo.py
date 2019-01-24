# !usr/bin/env python
# coding:utf-8

"""

author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/01/24
"""
import  numpy as np
import pandas as pd
from sklearn.cluster import KMeans

if __name__ == "__main__":
    print('Hello, Welcome to My World')
    X = np.loadtxt('data/20190107.txt').reshape(-1, 1)
    print(X[:5])
    k = 7
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300, tol=1e-6)
    y_pred = kmeans.fit_predict(X)
    print(kmeans.cluster_centers_)
    print(kmeans.n_iter_)
    print(pd.DataFrame(kmeans.labels_)[0].value_counts())