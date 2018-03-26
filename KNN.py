#!/usr/bin/python
#coding:utf-8

from numpy import *
import operator

def createDateSet():
	group = array([[1,0,1,1], [1,0,1,0], [0,0,0,0], [0,0,1,0]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inx, dataSet, labels, k):
    # shape 获得矩阵的行列数(行数，列数)
    dataSetSize = dataSet.shape[0]

    # tile 会将前面的inx矩阵放大(m,n)倍 例如：
    # >>> b = [1, 3, 5]
    # >>> tile(b, [2, 3])
    # array([[1, 3, 5, 1, 3, 5, 1, 3, 5],
    #        [1, 3, 5, 1, 3, 5, 1, 3, 5]])
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat**2
    # 求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号
    distances = sqDistances**0.5
    # 将距离从小到大排序，返回排好序的数组下标
    sortedDistIndicies = distances.argsort()

    # TOP K，统计分类出现的次数
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    # 降序 http://www.runoob.com/python/python-func-sorted.html
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print sortedClassCount
    return sortedClassCount[0][0]

group, labels = createDateSet()
print group
print labels
print classify0([0,0,1,2], group, labels, 3)