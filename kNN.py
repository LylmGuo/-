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


def file2matrix(filename): #读文件 返回其特征向量和标签向量
    with open(filename) as fr:
        arrayOlines = fr.readlines() #以换行符分隔，数据类型：list
        numberOfLines = len(arrayOlines) #被分成多少行
        returnMat = zeros((numberOfLines,3))  #建立行，列数量分别为numberOfLines，3的全为零的array
        classLabelVector = []
        index = 0
        for line in arrayOlines:
            line = line.strip()
            listFromLine = line.split("\t") #以tab符分隔特征，数据类型：list
            returnMat[index,:] = listFromLine[0:3]  #array[:,:]，逗号前表示y,逗号后表示x，a:b即从第a项到第b项。这里指第index行的所有值=listFromLine[0:3]这个list
            classLabelVector.append(int(listFromLine[-1]))
            index += 1

    return returnMat, classLabelVector
"""
file2matrix()步骤：
1. 读文件，获取文件行数
2. 建立一个全为零的array，行与文件行数量一致，列数量为特征数
3. 遍历文章行，以tab符分隔，前三项是特征，后一项是分类，分别存入两个list中
4. 返回按序排列的特征和分类
"""

#datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")


def autoNorm(dataSet): #归一化
    minVals = dataSet.min(0) #min(0)：取列的最小值，min(1)：取行的最小值；此处是获得每一列的最小值   type(minVals)=numpy.ndarray
    maxVals = dataSet.max(0) #最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #以dataSet的shape创建一个都为0的normDataSet
    m = dataSet.shape[0]  #获取dataSet的行
    normDataSet = dataSet - tile(minVals,(m,1))   #tile:复制minVals延y轴方向复制m倍，矩阵相减，减最小值得到该值到最小值的距离
    normDataSet = normDataSet/tile(ranges,(m,1))  #tile:复制ranges延y轴方向复制m倍，矩阵相除，除以范围，得到该点到最小值除以范围的值，作归一化
    return normDataSet, ranges, minVals

"""
autoNorm()步骤：
1. 输入dataSet，取dataSet每一列的最大值和最小值，相减，获得范围
2. 创建和dataSet相同shape（即行列数量一致）的normDataSet
3. 计算每一行的值到最小值的距离/范围，使得结果落到(0,1)之间，获得归一化的结果
4. 返回此归一化后的结果，范围和最小值
"""
