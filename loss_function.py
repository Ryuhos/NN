# Loss Function
# sse / cross entropy

import sys
sys.path
sys.path.append("D:/paperstudy/start/activation_function.py")
#sys.path 에 ActivationFunction을 불러오기 위해 path.append로 추가해준다.

import numpy as np
from activation_function import ActivationFunction

class LossFunction:
    def __init__(self,y,pred_y):
        self.y = y
        self.pred_y = pred_y

    def sse(self):
        result = np.sum((self.y-self.pred_y)^2)/2
        return result

    def cross_entropy(self):
        result = -np.sum(self.y*np.log(self.pred_y))
        return result

x = [[1,2,3,4],[6,8,9,10]]
w = [[1,5,2],[3,2,1],[3,5,1],[2,3,6]]
h = ActivationFunction(x,w)
np.dot(x,(h.h>0))
h.relu()