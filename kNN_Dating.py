from numpy import *
import operator
import matplotlib.pyplot as plt

"""
KNN(K临近算法)
Input: inX:被检验的输入向量   
       dataSet:输入的训练样本集   
       labels:训练样本中的标签向量   
       K:选择最近邻居的数目
Output: sortedClassCount[0][0]:k个距离最近的类中出现最多次的类
"""
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

#step1:prepare the data
#read the data from the file and put them in the matrices
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)          #get the numbers of lines
    returnMat = zeros((numberOfLines, 3))     #creat the same format matrix witl all 0
    classLabelVector = []
    index = 0
    for i in range(len(arrayOLines)):
    #if the first line of file is attribute, activate this line
    #for i in range(1, len(arrayOLines)):
        line = arrayOLines[i]
        line = line.strip()                        #remove '\n' on the right side
        listFromLine = line.split('\t')            #split on a '\t\ into list of substrings
        returnMat[index, :] = listFromLine[0:3]       #Feature Matrix
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#step2:analysis the data
#make some scatter plots of the data
def scatterplot():
    axislabel = ['free flying miles earned per year',
                 'percentage of time spent playing video games',
                 'liters of ice cream consumption per week']
    datingDataMat, datingDataLabel = file2matrix('data/datingTestSet.txt')
    # fig = plt.figure()  #create a board to plot
    # if we activate the parts with j,all the graphs will be ploted in the same board
    j = 1
    for k in range(1, 3):
        for i in range(0, 3 - k):
            plt.figure()
            # ax = fig.add_subplot(3, 1, j)  # divide the board into 3 lines 1 column
            plt.scatter(datingDataMat[:, i], datingDataMat[:, i + k],
                        15*array(datingDataLabel), 15*array(datingDataLabel))
            plt.xlabel(axislabel[i])
            plt.ylabel(axislabel[i + 1])
            plt.savefig('graphs/figure'+str(j))
            j += 1
    plt.show()

#step3:Data processing
#Normalize the data into the numbers between 0 and 1
def autoNorm(dataSet):
    minVals = dataSet.min(0)            #the (0) means we get the minVals of each column
    maxVals = dataSet.max(0)
    ranges = maxVals -minVals            #pay attention that ranges is not 'range'
    normDataSet = zeros(shape(dataSet))  #initialize the normal dataSet with all 0
    m = dataSet.shape[0]
    #newValue = (oldValue-minValue)/(maxValue-minValue)
    normDataSet =dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


#step4:Test the algorithm
#test the algorithm with KNN and get the error rate
def datingClassTest():
    hoRatio = 0.1                            #in the file we take 10% of it as a test group
    datingDataMat, datingDataLabel = file2matrix('data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    #in the file we take 90% of it as a train group
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i], normMat[numTestVecs:m],
                                     datingDataLabel[numTestVecs:m], 3)
        print('the classify came back with:%d, the real answer is %d'
              % (classifierResult, datingDataLabel[i]))
        if(classifierResult != datingDataLabel[i]):
            errorCount = errorCount + 1.0
    errorRate = errorCount/float(numTestVecs)
    print('the total error rate is :' + str(errorRate))

#step5:use the algorithm
#use the algorithm to predict if I'll like the person I date
def classifyPerson():
    print('Please enter the features of the person you are going to meet')
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(input("free flying miles earned per year:"))
    percentTgames = float(input("percentage of time spent playing video games:"))
    iceCream = float(input("liters of ice cream consumption per week:"))
    datingDataMat, datingDataLabel = file2matrix("data/datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTgames, iceCream])
    inArr = (inArr-minVals)/ranges
    classifierResult = classify0(inArr, normMat, datingDataLabel, 3)
    result = resultList[classifierResult-1]
    print("You will probably like this person :" + result)

#additional:To know the best percentage I take from data as tranning Set
#How do I divide the whole data set into trainning set and test set has minimum error rate?
def datingClassTest(testSetRatio):
    hoRatio = testSetRatio
    datingDataMat, datingDataLabel = file2matrix('data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i], normMat[numTestVecs:m],
                                     datingDataLabel[numTestVecs:m], 3)
        if(classifierResult != datingDataLabel[i]):
            errorCount = errorCount + 1.0
    errorRate = errorCount/float(numTestVecs)
    return errorRate

def ErrorRatePlot():
    testSetRatio = linspace(0.01, 0.99, 99)
    ErrorRate = []
    for i in range(len(testSetRatio)):
        ErrorRate.append(datingClassTest(testSetRatio[i]))
    plt.figure()
    plt.xlabel("The percentage of test set in the whole data set")
    plt.ylabel("The error rate of predict")
    plt.plot(testSetRatio, ErrorRate, color='red')
    plt.savefig('graphs/ErrorRate')
    plt.show()

#def errorRateplot():
#     testSetRatio = 0.01
#     plt.figure()
#     plt.xlabel("The percentage of test set in the whole data set")
#     plt.ylabel("The error rate of predict")
#     minTestSetRatio = 0
#     minErrorRate = 1
#     while(testSetRatio<1):
#         errorRate = datingClassTest(testSetRatio)
#         if(errorRate<minErrorRate):
#             minErrorRate = errorRate
#             minTestSetRatio =testSetRatio
#         plt.scatter(testSetRatio, errorRate, color='red')
#         testSetRatio += 0.01
#     #print('The best choice of test set is %f, with an minimum error rate of %f'
#     #      % (minTestSetRatio, minErrorRate))
#     plt.show()
#     #return minTestSetRatio, minErrorRate


if __name__ == '__main__':
    #scatterplot()
    classifyPerson()
    #ErrorRatePlot()