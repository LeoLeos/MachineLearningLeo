import bayes
from numpy import *
#
# listOPosts, listClasses = bayes.loadDataSet()   #加载每一句的词向量以及对应的类别：是否侮辱性语言
# print(listOPosts, listClasses)
# myVocabList = bayes.createVocabList(listOPosts)     #将所有句向量合并成一个不重复的列表
# print('myVocabList',myVocabList)
# trainMat = []
# for postinDoc in listOPosts:        #遍历每句词向量
#     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
#     print('postinDoc',postinDoc)
# print('trainMat',trainMat)
# print('listClasses',listClasses)
# p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)
# print(p0V,p1V,pAb)
# bayes.testingNB()
bayes.spamTest()