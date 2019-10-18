from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX,dataSet,labels,k):  #inX是需要分类的向量，可以是list或者tuple，dataSet是训练数据集特征向量，labels是标签向量，最后的参数k表示用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0]  #array的长度，即有多少行
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #tile() 复制inX延y轴方向复制dataSetSize倍,减 dataSet array 
    #输入向量的列和训练集的列一样多，行不一样多，于是此处需要复制inX复制成和训练数据集向量一样那么多
    sqDiffMat = diffMat ** 2 #相当于各求(x-inputX)和(y-inputY)的平方
    sqDistances = sqDiffMat.sum(axis=1)  #将上述平方两两相加 sum()中，参数axis=1表示按行相加 , axis=0表示按列相加
    distances = sqDistances ** 0.5 #开根号
    sortedDistIndicies = distances.argsort()  #argsort() 升序排列后返回原array的对应索引值
    classCount = {}
    for i in range(k):  #将sortedDistIndicies前k个的label记录下来
        voteIlabel = labels[sortedDistIndicies[i]]  
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1   #get方法，前者是需要在字典中寻找的参数，后者是若找不到赋值的参数；找到后进行+1操作，找不到先赋值0再进行+1操作
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)  #key = operator.itemgetter(1)--按字典的第二个元素排列，对出现的label按出现次数降序排序
    return sortedClassCount[0][0]  #返回降序排序第一位的值，即出现最多次的label的值
    
"""
classify0()步骤：
1. 允许输入值：需要进行分类的向量，可以是list或者tuple；训练数据集特征向量；训练数据集标签向量，最近邻居的数目（参数k）
2. 计算训练数据集特征向量的长度
3. 计算 （需要进行分类的向量 分别减 每行训练数据集特征向量）x,y各自平方后相加 再开根号，得到 需要进行分类的向量 距离 训练数据集特征向量 中每个点的距离。
4. 按距离（从小到大）升序排列后，返回对应原array的索引值
5. 取出排列后的向量对应的标签向量
6. 计算每个标签向量出现的次数
7. 返回出现次数最多的标签向量
"""


#group, labels = createDataSet()
#classify0([0,0], group, labels, 3)
