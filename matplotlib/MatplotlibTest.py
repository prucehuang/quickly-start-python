#!/usr/bin/python
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

x,y = np.loadtxt("D://smoba.xls", delimiter=',', usecols=(0, 1), unpack=True)
# a = np.loadtxt("D://smoba.xls", delimiter=',', usecols=(0,1))
# print type(a)
# print len(a)
# print a[0]
print(len(x))
print(len(y))
print(type(x))
# red dashes, blue squares and green triangles
plt.plot(x, y, '.')
plt.show()