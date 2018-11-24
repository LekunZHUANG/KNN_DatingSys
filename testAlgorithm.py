from numpy import *

def testshape():
    a = array([[1, 6, 7],
               [8, 7, 2],
               [6, 5, 8],
               [1, 5, 7]])
    print(a.shape)  # shape 获取数组形状
    print(a.shape[0])  # shape[0] 获取数组行数
    print(a.shape[1])  # shape[1] 获取数组列数

def testtile():
    print(tile([2, 3], 4))  # 将数组[2,3] 扩展成一行四列
    print(tile([2, 3], (4, 1)))  # 将数组[2,3] 扩展成四行一列
    print(tile([2, 3], (1, 4)))  # 将数组[2,3] 扩展成一行四列
    print(tile([2, 3], (3, 4)))  # 将数组[2,3] 扩展成三行四列

def testsumaxis():
    a = array([[1, 6, 7],
               [8, 7, 2],
               [6, 5, 8],
               [1, 5, 7]])
    intx = array([3, 5, 5])
    diffMat = tile(intx, (4, 1)) - a
    # print(diffMat)
    sqDiffMat = diffMat ** 2
    #print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=0)  # axis=0表示对列求和  返回一个一行三列的数组
    #print(distances)
    #print(distances.shape)
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1表示对行求和  返回一个四行一列的数组
    #print(distances)
    #print(distances.shape)
    #print(distances[3, ])
    distances = sqDistances**0.5
    return distances

def testargsort():
    a= array([2.5, 3.6, 1.2, 8.5, 4.3, 9.7, 5.2])
    sa = a.argsort()
    print(sa)

def test():
    distances = testsumaxis()
    print(distances)
    sortedDistIndicies = distances.argsort()  # argsort返回数组排序后的下标索引
    print(sortedDistIndicies)
    # print(sortedDistIndicies[0])
    labels = ['male', 'female', 'male', 'male']
    classCount = {}
    voteIlabel = labels[sortedDistIndicies[0]]
    print(voteIlabel)
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    print(classCount)
    voteIlabel = labels[sortedDistIndicies[1]]
    print(voteIlabel)
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    print(classCount)
    voteIlabel = labels[sortedDistIndicies[2]]
    print(voteIlabel)
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    print(classCount)

test()
