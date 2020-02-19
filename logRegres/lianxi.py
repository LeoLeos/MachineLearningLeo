import logRegres
from numpy import *
dataArr,labelMat=logRegres.loadDataSet()
#批处理梯度上升权重
gradAscent_weight=logRegres.gradAscent(dataArr,labelMat)
logRegres.plotBestFit(gradAscent_weight.getA())

# #随机梯度上升权重
# weights_stocGradAscent0=logRegres.stocGradAscent0(array(dataArr),labelMat)
# logRegres.plotBestFit(weights_stocGradAscent0)
# #随机梯度上升权重改进版
# weights_stocGradAscent1=logRegres.stocGradAscent1(array(dataArr),labelMat)
# logRegres.plotBestFit(weights_stocGradAscent1)
# logRegres.multiTest()
