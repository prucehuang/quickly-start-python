#!/usr/bin/python
#coding:utf-8
# https://github.com/lijin-THU/notes-python/blob/master/12-pandas/12.01-ten-minutes-to-pandas.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pandas 中有三种基本结构：
print('一、Series：1D labeled homogeneously-typed array')
s = pd.Series([1,3,5,np.nan,6,8])
print(s)
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64
print(s[1])
# 3.0

print('二、DataFrame：General 2D labeled, size-mutable tabular structure with potentially heterogeneously-typed columns')
dates = pd.date_range('20130128', periods=6)
print(dates)
# DatetimeIndex(['2013-01-28', '2013-01-29', '2013-01-30', '2013-01-31',
#                '2013-02-01', '2013-02-02'],
#               dtype='datetime64[ns]', freq='D')
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)
#                    A         B         C         D
# 2013-01-28 -1.192827  0.226132  0.392934 -0.826371
# 2013-01-29  0.257592  0.401018 -0.227618 -0.288590
# 2013-01-30  1.761285  1.251978 -0.817694  0.729408
# 2013-01-31 -1.373272  0.488828 -1.934463  2.471025
# 2013-02-01  0.815091  0.940244  0.570265 -0.568387
# 2013-02-02 -0.545106 -0.360151 -0.124949 -0.496492
print(df.describe())

print('三、Panel：General 3D labeled, also size-mutable array')


print('END')
