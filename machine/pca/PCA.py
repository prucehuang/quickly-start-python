# !/usr/bin/python
# coding:utf-8

'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    # 计算协方差 将(m, n) 的数据压缩成 (n, n)
    covMat = cov(meanRemoved, rowvar=0)
    print('covMat : ', covMat.shape)
    # 计算矩阵的特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    print(covMat)
    print(eigVals)
    print(eigVects)
    print(covMat*eigVects)
    print(eigVals*eigVects.A)
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def pcaAnalysis():
    dataMat = loadDataSet('data\\testSet.txt')
    print(dataMat.shape)
    lowDMat, reconMat = pca(dataMat, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

# 将NAN的数据替换成Mean，减少误差
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

if __name__ == "__main__":
    pcaAnalysis()