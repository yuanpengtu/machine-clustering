# -*- coding: utf-8 -*-
# __Author__: Sdite
# __Email__ : a122691411@gmail.com

import pandas as pd
import matplotlib.pyplot as plt
from numpy import shape

if __name__ == '__main__':

    # iris鸢尾花数据集包含3个不同品种的鸢尾花（Setosa, Versicolour, and Virginica）数据, 花瓣和萼片长度, 存储在一个150*4的 numpy.ndarry中
    # 150行4列, 150行指150朵花, 4列分别是Sepal Length, Sepal Width, Petal Length and Petal Width

    # 数据清洗
    # 只取萼片的长度和宽度 来测试 我们的k-means
    dataSet = pd.read_csv('iris.data')
    dataMat = dataSet.loc[:, ['sepal-length', 'sepal-width']]
    dataMat.to_csv('iris.csv', index=None)


    # 以下是画出真实实际的分类

    dataSet[dataSet['class'] == 'Iris-setosa'] = 0
    dataSet[dataSet['class'] == 'Iris-versicolor'] = 1
    dataSet[dataSet['class'] == 'Iris-virginica'] = 2
    labels = dataSet.loc[:, ['class']]

    m, n = shape(dataSet)
    markSamples = ['dr', '*b', 'sg', 'pk', '^r', '+b', '<g', 'hk']
    label = ['0', '1', '2']
    # 通过循环的方式, 完成分组散点图的绘制
    for i in range(m):
        plt.plot(dataMat.iat[i, 0], dataMat.iat[i, 1], markSamples[int(labels.iat[i, 0])], markersize=5)

    for i in range(0, m, 50):
        plt.plot(dataMat.iat[i, 0], dataMat.iat[i, 1], markSamples[int(labels.iat[i, 0])], markersize=10,
                 label=label[int(labels.iat[i, 0])])
    plt.legend(loc='upper left')
    plt.title('True Classification')
    plt.show()


