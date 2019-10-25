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

  
