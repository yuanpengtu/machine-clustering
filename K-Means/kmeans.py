# -*- coding: utf-8 -*-
# __Author__: Sdite
# __Email__ : a122691411@gmail.com

import pandas as pd
import matplotlib.pyplot as plt
from numpy import sum, power, shape, zeros, mat, min, max, random, nonzero, inf, mean

BEGINING = 0            # 最开始状态
CLASSIFICATION = 1      # 分类
NEWCENTROID = 2         # 新的聚类中心

# 计算两个向量间的距离, 返回值是算术平方根的平方
def calcDistance(vecA, vecB):
    return sum(power(vecA - vecB, 2))

# 构建随机的k-means的k个质心
def randCentroid(data_set, k):
    n = shape(data_set)[1]           # 获取数据的列数 n
    centroids = mat(zeros((k, n)))    # 初始化k行n列矩阵, 表示k个质心的向量

    # 把质心初始化在数据范围中
    # 即 比如一个二维数据, n就为2
    # 下面的for循环第一次, 就是先得到横坐标的最小值minJ和最大值maxJ
    # 然后求出横坐标最大值和最小值的差值rangeJ
    # 那么质心的横坐标就等于 minJ + rangeJ*一个百分比
    # 由此完成一个随机质心的初始化
    for j in range(n):
        minJ = min(data_set[:, j])   # 矩阵第j列中的最小值
        maxJ = max(data_set[:, j])   # 矩阵第j列中的最大值
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)

    return centroids

def kMeans(dataSet, k):
    m = shape(dataSet)[0]                   # 获取数据的行数

    # 用于存放样本属于哪个类以及这到这个类质心的距离
    # cluster_assessment的第一列存放所属的中心点,
    # 第二列存放到质心的距离
    clusterAssessment = mat(zeros((m, 2)))

    centroids = randCentroid(dataSet, k)   # 随机k个质心

    # 返回产生的聚类中心
    yield centroids, clusterAssessment, BEGINING

    clusterConvergence = False              # 用于判断聚类是否收敛
    while not clusterConvergence:
        clusterConvergence = True

        # 循环求出每个点到各个质心的距离
        for i in range(m):
            minDist = inf
            minIndex = -1

            # 计算第i个点到质心的距离
            for j in range(k):
                distJI = calcDistance(centroids[j, :], dataSet[i, :])

                if distJI < minDist:
                    # 说明i到j更近，更新质心和到质心的距离
                    minDist = distJI
                    minIndex = j

            if clusterAssessment[i, 0] != minIndex:
                # 质心有变化, 说明还没收敛
                clusterConvergence = False
                clusterAssessment[i, :] = minIndex, minDist

        # 返回聚类的结果
        yield centroids, clusterAssessment, CLASSIFICATION

        # 重新计算质心
        for cent in range(k):
            # 取出是k类的元素
            ptsInCluster = dataSet[nonzero(clusterAssessment[:, 0] == cent)[0]]
            # 如果没有元素分到这个聚类中, 则没必要重新计算新的质心
            if ptsInCluster.size:
                # 更新新的质心
                centroids[cent, :] = mean(ptsInCluster, axis=0)

        # 返回产生新的聚类中心
        yield centroids, clusterAssessment, NEWCENTROID


# 设置最多分八个聚类
mark = ['dr', '*b', 'sg', 'pk', '^r', '+b', '<g', 'hk']

if __name__ == '__main__':
    dataSet = mat(pd.read_csv("iris.csv"))
    k = 4
    for centroids, clusterAssessment, status in kMeans(dataSet, k):
        centroidsX = centroids.getA()[:, 0]
        centroidsY = centroids.getA()[:, 1]

        legends = []
        for i in range(k):
            legend, = plt.plot(centroidsX[i], centroidsY[i], mark[i], markersize=10)
            legends.append(legend)
        plt.legend(legends, list(range(k)), loc="upper left")

        if status == CLASSIFICATION:
            for i in range(len(dataSet)):
                plt.plot(dataSet[i, 0], dataSet[i, 1], mark[int(clusterAssessment[i, 0])], markersize=5)
        elif status == BEGINING:
            # 最开始状态没有分类, 所以未分类的颜色点颜色都是一样的
            for i in range(len(dataSet)):
                plt.plot(dataSet[i, 0], dataSet[i, 1], 'oc', markersize=5)

        plt.show()

