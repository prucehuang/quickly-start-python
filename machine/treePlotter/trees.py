# !/usr/bin/python
# coding:utf-8
# ***************************************************************
# 决策树
# author:   pruce
# email:    1756983926@qq.com
# date:     2018年4月19日
# ***************************************************************

from math import log
import operator

# 计算一个数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

# 用value筛选axis下标对应的数据
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # list[m:n]表示的是[m, n)左闭右开区间
            reducedFeatVec = featVec[: axis]
            # 将多个值扩展到list
            reducedFeatVec.extend(featVec[axis + 1:])
            # 将一个值追加到list
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选出最优的划分决策树字段——列号
def chooseBestFeatureToSplit(dataSet):
    # 获得列数, 最后一列是label, 每一列表示一个特征的全部取值
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 获得原始数据的第i列数据值
        # example表示一行数据
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 乘上prob是为了归一化
            newEntropy += prob * calcShannonEnt(subDataSet)

        print i, '列: ', newEntropy
        # bestInfoGain表示熵减少的最大值
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print chooseBestFeatureToSplit(dataSet)