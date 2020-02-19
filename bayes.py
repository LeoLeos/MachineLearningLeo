'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 标记为侮辱性语言,0标记为不是
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #对向量的每一维合并一个不重复的数组
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec    #返回词汇列表的每个词汇是否出现在数据集当中，如果是则标记为1

def trainNB0(trainMatrix,trainCategory):#trainMatrix文档矩阵，trainCategory每篇文档类别标签构成的向量
    numTrainDocs = len(trainMatrix)     #len获取行数
    print('numTrainDocs',numTrainDocs)
    numWords = len(trainMatrix[0])      #第一条词向量的数目
    pAbusive = sum(trainCategory)/float(numTrainDocs)       #侮辱性语言占总词条的比例
    p0Num = ones(numWords); p1Num = ones(numWords)      #都设为1是防止相乘之后为0，减少影响
    print(p0Num)
    p0Denom = 2.0; p1Denom = 2.0                        #由于之前设为1了，故此处设为2
    for i in range(numTrainDocs):       #遍历文档里每一句词向量
        if trainCategory[i] == 1:       #若1则说明此词向量为侮辱性语言
            p1Num += trainMatrix[i]     #对所有侮辱性语言的词向量相加
            p1Denom += sum(trainMatrix[i])      #获取这一句单词个数，进行累加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive        #侮辱性语言占总词条的比例

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #vec2Classify为该词条在词汇表集出现的标记为1的向量，p0Vec为0类词汇的概率向量，p1Vec为1类词汇的概率向量，pClass1类别为1占全部词条的比例
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

#词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec        #返回inputSet在vocabList各个出现的个数的矩阵


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc',thisDoc)
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc',thisDoc)
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)     #侮辱性语言标记为1
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)     #非侮辱性语言标记为0
    # print('docList',docList)
    # print('classList',classList)
    # print('fullText',fullText)
    vocabList = createVocabList(docList)#创建词汇表
    trainingSet = list(range(50)); testSet=[]           #初始化训练集和测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))     #随机产生下标
        testSet.append(trainingSet[randIndex])      #加到测试集
        del(trainingSet[randIndex])     #删除作为测试集的元素
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#对训练集遍历
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))#在训练矩阵加入各词条的词包
        trainClasses.append(classList[docIndex])        #加入对应的类别
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))       #训练算法之后返回参数
    errorCount = 0      #初始化错误个数
    #交叉验证
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#预测类别
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):#fullText词向量，vocabList全部词的汇总
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)   #计算各个次次数
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) #按词出现频率从大到小排序
    return sortedFreq[:30]      #返回前三十频率最高

def localWords(feed1,feed0):
    import feedparser
    #与前个函数相同的步骤
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))       #训练获取参数
    errorCount = 0
    for docIndex in testSet:        #分类验证
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
