# !/usr/bin/python
# coding:utf-8
# ***************************************************************
# 实现一个分类器，使用KNN的方法对图像文件分类
# 训练数据data/trainingDigits；测试数据data/testDigits
#
# author:   pruce
# email:    1756983926@qq.com
# date:     2018年4月18日
# ***************************************************************

from numpy import *
from os import listdir
import operator

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

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
    return sortedClassCount[0][0]

def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('data/trainingDigits')
    m = len(trainingFileList)
    # 将32*32的图像拍平
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        trainingMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

    testFileList = listdir('data/testDigits')
    errorCount = 0.0
    m = len(testFileList)
    for i in range(m):
        fileNameStr = testFileList[i] # 0_13.txt
        vectorUnderTest = img2vector('data/testDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 2)
        fileStr = fileNameStr.split('.')[0]  # 0_13
        classNumStr = int(fileStr.split('_')[0])  # 0
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1

    print "\nthe total number of errors is: %d, error rate is %f" %(errorCount, errorCount/float(m))

handWritingClassTest()