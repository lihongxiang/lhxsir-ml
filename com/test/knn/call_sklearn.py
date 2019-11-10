# -*- encoding: utf-8 -*-
"""
call_sklearn.py
Created on 2019/11/10 0010 下午 4:33
@author: LHX
调用sklearn实现KNN算法
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343853454, 3.368312451],
              [3.582294121, 4.679917921],
              [2.280362211, 2.866990212],
              [7.423436752, 4.685324231],
              [5.745231231, 3.532131321],
              [9.172112222, 2.511113104],
              [7.927841231, 3.421455345],
              [7.939831414, 0.791631213]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] # 设置训练组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y) # 将数据可视化

x=np.array([8.093607318,3.365731514])

# 创建kNN_classifier实例
kNN_classifier = KNeighborsClassifier(n_neighbors=6)
# kNN_classifier做一遍fit(拟合)的过程，没有返回值，模型就存储在kNN_classifier实例中
kNN_classifier.fit(X_train, y_train)
# kNN进行预测predict，需要传入一个矩阵，而不能是一个数组
y_predict = kNN_classifier.predict(x.reshape(1,-1))
print(y_predict)

