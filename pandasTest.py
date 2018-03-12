#!/usr/bin/python
#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pandas 中有三种基本结构：
print '一、Series：1D labeled homogeneously-typed array'
s = pd.Series([1,3,5,np.nan,6,8])
print s
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64
print s[1]
# 3.0

print '二、DataFrame：General 2D labeled, size-mutable tabular structure with potentially heterogeneously-typed columns'


print '三、Panel：General 3D labeled, also size-mutable array'


print 'hello'