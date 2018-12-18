# !/usr/bin/python
# coding:utf-8
# ***************************************************************
# 无监督学习，KMeans的实现
# 训练数据data/trainingDigits
# k Means Clustering for Ch10 of Machine Learning in Action
# author: Peter Harrington
# email:    1756983926@qq.com
# date:     Created on Feb 16, 2011
# ***************************************************************

from numpy import *

def loaddataMat(fileName):      #general function to parse tab -delimited floats
    dataSet = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine) #map all elements to float()
        dataSet.append(fltLine)
    return mat(dataSet)

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

# 真的是随机选K个点
def randCent(dataSetMat, k):
    n = shape(dataSetMat)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSetMat[:,j]) 
        rangeJ = float(max(dataSetMat[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
    
# 随机选取质心 进行迭代 直到没有点的质心改变为止
def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf # 正无穷 
            minIndex = -1
            # 获得距离最近的一个核心minDist, minIndex
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataMat[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            # 如果还有点的最新质心有改变的话就继续迭代
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):#recalculate centroids
            # .A 将矩阵转为数组
            # get all the point in this cluster
            # nonzero() 选出非0的下标
            # https://blog.csdn.net/u013698770/article/details/54632047
            # clusterAssment[:,0].A==cent 判断每个点的质心是不是为cent
            ptsInClust = dataMat[nonzero(clusterAssment[:,0].A==cent)[0]]
            # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
            # axis = 1：压缩列，对各行求均值，返回 m *1 矩阵
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

def analysisModel(clusterAssment):
    k = int(max(clusterAssment[:, 0].A)[0]) + 1
    # 打印族的属性
    centDistList = {}
    centCountList = {}
    for i in range(k):
        centDistList[i] = 0
        centCountList[i] = 0
    for assment in clusterAssment.A:
        cent = int(assment[0])
        dist = assment[1]
        centDistList[cent] += dist
        centCountList[cent] += 1
    for i in range(k):
        print(i, centCountList[i], centDistList[i])

# 二分K均值聚类算法 每次只将一个族一分为二 
def biKmeans(dataMat, k, distMeas=distEclud):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    # 初始化将所有的点归为0族,并且计算它们的距离
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataMat[j,:])**2
        
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                # 这里存了两个新族的质心
                bestNewCents = centroidMat
                # 这里只存了两个新生的族0、1
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将1族的族号设置为当前的族数
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        # 将i族的族号继承给0号族
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 追加新的质点 更新旧族的质点
        centList.append(bestNewCents[1,:].tolist()[0])
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        # 更新旧最佳切割质点的全部点的质心
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss #reassign new clusters, and SSE
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('D:\Document\quickly-start-python\machine\KMeans\data\\places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('D:\Document\quickly-start-python\machine\KMeans\data\\Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
