# !/usr/bin/python
# coding:utf-8
# ***************************************************************
# 实现一个分类器，使用KNN的方法
# 读文件数据 - 数据结构可视化
# 读文件数据 - 数据归一化 - 分类器 - 测试分类器的准确度
# author:   pruce
# email:    1756983926@qq.com
# date:     2018年4月18日
# ***************************************************************

from numpy import *
import operator
import matplotlib.pylab as plt

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()

    numberOfLines = len(arrayOlines)
    # 文本数据矩阵
    returnMat = zeros((numberOfLines, 3))
    # 数据label结果
    classLabelVector = []
    index = 0

    for line in arrayOlines :
        line = line.strip()
        listFromLIne = line.split('\t')
        returnMat[index, :] = listFromLIne[0: 3]
        classLabelVector.append(int(listFromLIne[-1]))
        index += 1

    return returnMat,classLabelVector

def showDataSet(dataSet, datingLabels):
    # 画图看一下数据
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 1], dataSet[:, 0], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

def autoNorm(dataSet):
    # 数据归一化，将每一列的数据值变成[0, 1]
    # 计算公式是 (value-min) / (max-min)

    # 获得每列的最小值、最大值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # shape 返回[m, n]
    m = dataSet.shape[0]
    # tile会按照给定的数据结构构件矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet

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

def datingClassTest():
    datingDataMat, datingLabels = file2matrix('data\datingTestSet.txt')
    normMat = autoNorm(datingDataMat)

    m = normMat.shape[0]
    hoRatio = 0.10
    numTestVecs = int(m * hoRatio)
    # 将前百分之十作为测试数据
    errorCount = 0.0
    for i in range(numTestVecs):
        # python的矩阵语法使用起来很是奥妙
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs: m], 3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1
    print "the total error rate is: %f" %(errorCount / float(numTestVecs))

datingClassTest()