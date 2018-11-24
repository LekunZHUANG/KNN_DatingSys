from numpy import *
import operator

def createDataSet():
    group = array([[1, 1], [2, 2], [2, 3], [3, 2], [3, 3], [4, 3]])
    f = 'female'
    m = 'male'
    labels = [f, f, m, f, m, m]
    return group, labels

#KNN(K临近算法)
#inX:被检验的输入向量   dataSet:输入的训练样本集   labels:训练样本中的标签向量   K:选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet      #获取检验点与所有点的差矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                  #差矩阵平方在行方向上求和
    distances = sqDistances**0.5                         #计算距离
    sortedDistIndicies = distances.argsort()             #获取排序后的下标数组
    classCount ={}
    for i in range(k):                                   #将k个距离最小点的标签,出现次数写入字典
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] =classCount.get(voteIlabel, 0) + 1
    #字典排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

group,labels =createDataSet()
classify0([2.7, 2.2], group, labels, 3)