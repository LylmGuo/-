from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def calcShannonEnt(dataSet): #计算某个dataSet的熵
    numEntries = len(dataSet) #获取实例总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] #dataSet中每行的最后一个特征/属性为Label，存为currentLabel
        if currentLabel not in labelCounts.keys(): #如果不在字典里的key中
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 #循环计算每个Label出现次数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries #计算概率 出现次数/整体数量
        shannonEnt -= prob * log(prob,2) #按公式计算熵，for循环实现相减（相当于相加后取负）
    return shannonEnt

"""
calcShannonEnt()步骤：
1. 获取实例总数
2. 循环将数据集中特征数据列取出，若该键其不曾在字典中出现，则将其加入labelCounts字典中，值设置为0
3. 每次出现一次，字典中相应的值+1
4. 循环字典，用该键出现次数/数据集长度来计算每个特征出现概率，计算每个特征的熵，求和取负
5. 返回数据集的熵
"""

  
def splitDataSet(dataSet,axis,value):  #待划分的数据集，用于划分数据集的特征列下标，我们想返回特征的值
    retDataSet = []
    for featVec in dataSet: 
        if featVec[axis] == value: #featVec [1, 1, 'yes'] axis=0：如果这一行第一列的值和value一致，进行以下操作
            reducedFeatVec = featVec[:axis] #获取这行第0到第axis-1列的列表
            reducedFeatVec.extend(featVec[axis+1:]) #reducedFeatVec加上featVec后面的列
            retDataSet.append(reducedFeatVec) #然后放到retDataSet列表中
    return retDataSet #返回的是除去（用来做划分数据集的特征）的数据集

"""
splitDataSet()步骤：
1. 循环查看数据集，选用于划分数据集的特征列，对比其值是否与我们想返回的特征的值相同
2. 若相同，删除该特征列，保留剩余的特征列和分类列（即“yes”和“no”那一列）
3. 存入新的数据集中，循环结束后返回新的数据集

"""
def chooseBestFeatureToSplit(dataSet): #dataSet每一行的最后一列必须为类型
    numFeatures = len(dataSet[0]) - 1  #计算减去类型后共有多少个特征
    baseEntropy = calcShannonEnt(dataSet) #计算原有的dataSet的熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #把第i列的数据提取出来，放到一个list中
        uniqueVals = set(featList) #去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value) #循环用去重后的每一个结果分别来划分dataSet（i与每个value匹配；与value相同的作为子集返回）
            prob = len(subDataSet)/float(len(dataSet)) #计算子集出现概率
            newEntropy += prob * calcShannonEnt(subDataSet) #计算熵
        infoGain = baseEntropy - newEntropy #计算信息增益，即旧熵-新熵
        if infoGain > bestInfoGain: #信息增益比较
            bestInfoGain = infoGain
            bestFeature = i #若满足条件 最好的特征就是第i个特征
    return bestFeature
#循环，遍历特征；遍历到某特征后，假设按其划分，要循环计算子集中不同种类（即不同value）的信息熵和信息增益；综上，来确定哪个特征来分类最好
"""
chooseBestFeatureToSplit()步骤：
1. 计算有多少个特征（dataSet中除去最后一列都是特征列）
2. 通过calcShannonEnt()函数计算dataSet的熵
3. 设初始最好信息增益为0，特征列下标为-1
4. 循环numFeatures（特征列数）次，将该列的数据提取出来并去重，存为uniqueVals
5. 循环uniqueVals的值，以其作为区分值，传给splitDataSet()进行数据集划分，存为子集
6. 计算子集出现概率
7. 计算子集的熵
8. 计算信息增益（旧熵-新熵）
9. 比较信息增益和最好信息增益，若本次信息增益较大，则存为最好信息增益，该特征存为最优特征
10. 循环结束后，返回最优特征
"""
