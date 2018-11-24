from numpy import *

fr = open('datingTestSet.txt')
arrayOLines = fr.readlines()
#print(arrayOLines)   #打印出来的\t相当于tab键
numberOfLines = len(arrayOLines)
#print(numberOfLines)
returnMat = zeros((numberOfLines, 3))
classLabelVector = []
index = 0
line = arrayOLines[0]
#print(line)
line = line.strip()
#print(line)
listFromLine = line.split('\t')
#print(listFromLine)
returnMat[index] =listFromLine[0:3]
#print(returnMat[index])
classLabelVector.append(int(listFromLine[3]))
#print(classLabelVector)