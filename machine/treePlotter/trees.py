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


# -------------------
# 从classList中选出现频次最大的class
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # operator.itemgetter(1) 使用classcount的第一类数据降序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 当classList[0]在classList中出现的次数等于classList的size的时候 classList里面只有一个元素
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree 
    
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 用决策树分类
def classify(inputTree, featLabels, testVec):
    # inputTree such as
    # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 找到根节点对应的特征值
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def createLensesDataSet(fileName, delim='\t'):
    fr = open(fileName)
    dataSet = [line.strip().split(delim) for line in fr.readlines()]
    labels = ['age', 'prescripy', 'astigmatic', 'tearRate']
    return dataSet, labels

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print chooseBestFeatureToSplit(dataSet)