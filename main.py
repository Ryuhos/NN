# main
import sys
sys.path.append('D:/paperstudy/start/optimizer.py')
sys.path.append('D:/paperstudy/start/layers.py')

import numpy as np
import pandas as pd
from optimizer import Optimizer
from layers import make_w

# iris data 불러오기
data = pd.read_csv("iris.csv")
x = data.values[:,:4]
y = data.values[:,4]
y_onehot = np.array(pd.get_dummies(y))
W = make_w(x,y_onehot)

model1 = Optimizer(x,y,W)
model1.Gradient_relu(epochs=100,lr=0.05)