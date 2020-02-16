import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *


group, labels = kNN.createDataSet()
#print(group, labels)
result = kNN.classify0([0,0],group,labels,3)
#print(result)
datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
#print(returnMat,classLabelVector)
#实例化画图视图
fig = plt.figure()
#选择子图
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
#显示画图结果
#plt.savefig('./1.jpg')
#plt.show()
#归一化
normDataSet, ranges, minVals = kNN.autoNorm(datingDataMat)
kNN.datingClassTest()
kNN.classifyPerson()