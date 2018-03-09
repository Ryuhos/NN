# layer 일반화 #

import sys
sys.path
sys.path.append('D:/paperstudy/start/activation_function2.py')

import numpy as np
from activation_function2 import ActivationFunction


# x , y , w의 shape를 받으면 W를 생성하는 함수
def make_w(x,y,shape_w = [15,10]):
    form = [np.array(x).shape[1]]
    for i in shape_w:
        form.append(i)
    form.append(np.array(y).shape[1])
    W = []
    result_w = []
    for i in range(len(form)-1):
        W.append([form[i], form[i + 1]])
    for j in range(len(W)):
        a = np.random.randn(W[j][0],W[j][1])+1
        result_w.append(a)
    return result_w

class layers:
    def __init__(self,x,y,w):
        self.x = np.array(x)
        self.y = np.array(y) #onehot으로 들어와야 된다.
        self.w = w

    def layer_relu(self):
        H = self.x
        self.H = []
        self.J = []
        self.diff = []
        for i in range(len(self.w)):
            J = np.dot(H,self.w[i])
            self.J.append(J)
            K = ActivationFunction(J)
            if i != len(self.w)-1:
                H = K.relu()
                self.H.append(H)
            else :
                self.O = K.softmax()
            self.diff.append(np.float16(K.back))
        self.H.append(K.back)
        self.pred_y = np.argmax(self.O, axis=1)
        return self.O

    def layer_sigmoid(self):
        H = self.x
        self.H = []
        self.J = []
        self.diff = []
        for i in range(len(self.w)):
            J = np.dot(H,self.w[i])
            self.J.append(J)
            K = ActivationFunction(J)
            if i != len(self.w)-1:
                H = K.sigmoid()
                self.H.append(H)
            else :
                self.O = K.softmax()
            self.diff.append(np.float16(K.back))
        self.H.append(K.back)
        self.pred_y = np.argmax(self.O, axis=1)
        return self.O

