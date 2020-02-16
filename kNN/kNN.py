'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      #tile产生以inX为元素包，形成(dataSetSize,1)矩阵，返回inX与dataSet各点坐标差值。
    sqDiffMat = diffMat**2      #计算点inX与各点欧式距离平方。
    sqDistances = sqDiffMat.sum(axis=1)     #以行为点，对列求和
    distances = sqDistances**0.5        #开方
    sortedDistIndicies = distances.argsort()        #从小到大排序的下标
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]      #获取按sortedDistIndicies排序的label
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1       #获取与距离最小的前k个分给voteIlabel类有几次
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)     #按分给某一类的次数排序
    return sortedClassCount[0][0]       #返回分给次数最多的label

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #获取文件行数
    returnMat = zeros((numberOfLines,3))        #构建零矩阵
    classLabelVector = []                       #构建准备存储label向量
    fr = open(filename)
    index = 0
    #将文件转化为矩阵
    for line in fr.readlines():
        #去掉每行首尾空格回车符
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]

        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #产生与输入矩阵一样大小的零矩阵
    normDataSet = zeros(shape(dataSet))
    #获取行数
    m = dataSet.shape[0]
    #对输入矩阵减去最小值
    normDataSet = dataSet - tile(minVals, (m,1))
    #除以maxVals 与 minVals的差值
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    #返回归一化之后的矩阵，最大值与最小值的差以及最小值
    return normDataSet, ranges, minVals
   
def datingClassTest():
    #设置测试样本与样本数比例
    hoRatio = 0.50      #hold out 10%
    #加载数据
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    #归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #设置k值
    numTestVecs = int(m*hoRatio)
    #初始化错误率
    errorCount = 0.0
    for i in range(numTestVecs):
        #判别normMat[i,:]测试样本的类别
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
    
def img2vector(filename):
    #初始化零向量矩阵
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            #按行，列写入像素数值
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect       #返回图片向量

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set加载训练集
    m = len(trainingFileList)       #训练集样本数
    trainingMat = zeros((m,1024))       #为美国样本初始化一个零向量，以存储每个样本的向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set加载测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)     #将测试集的每个样本逐步向量化
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)     #为其分类
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0     #计算分错类的个数
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))     #错误率

def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(input('percentTage of time spent playing video game?'))
    ffMiles = float(input('frequent filer miles earned per years?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges, minVals = autoNorm(datingDataMat)
    #将三者转化成数组，之前打错成大括号，变成列表，报错！
    inArr = array([ffMiles, percentTats, iceCream])
    #调用kNN函数进行预测
    classifierResult = classify0((inArr - minVals)/ranges,normMat, datingLabels, 3)
    print('You will probably like this person:',resultList[classifierResult - 1])

