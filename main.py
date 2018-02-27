# main

import numpy as np
import pandas as pd
# import os
# os.getcwd()

# iris data 불러오기
data = pd.read_csv("iris.csv")
x = data.values[:,:4]
y = data.values[:,4]

# w1 shape(#column_of_x,hidden1_node) 생성
# w2 shape(hidden1_node,hidden2_node) 생성
# w3 shape(hidden2_node,y_label) 생성

# h1 = np.dot(x,w1)
# H1 = ActivationFunction(h1)
# H1 = H1.sigmoid()
# h2 = np.dot(H1,w2)
# H2 = ActivationFunction(h2)
# H2 = H2.sigmoid()
# h3 = np.dot(H2,w3)
# y_hat = ActivationFunction(h3)
# y_hat = y_hat.softmax()
# loss = LossFunction(y,y_hat)
# loss = loss.cross_entropy()

# optimizer = optimizer(layer)
# optimizer = optimizer.Gradient()
# 이거 layer 를 받아들이게 만들면 일반화가 되겠네 그러면 layer 클래스를 만들어 보자

# 결과값은 w = [] , loss
